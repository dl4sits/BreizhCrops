import os

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np


from .urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, FILESIZES, SHP_URLs, H5_URLs, RAW_CSV_URL
from ..utils import download_file, unzip, untar

BANDS = {
    "L1C": ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa'],
    "L2A": ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B11', 'B12', 'CLD', 'EDG', 'SAT','doa']
}

PADDING_VALUE=-1

class BreizhCrops(Dataset):

    def __init__(self,
                 region,
                 root="breizhcrops_dataset",
                 year=2017, level="L1C",
                 transform=None,
                 target_transform=None,
                 filter_length=0,
                 verbose=False,
                 load_timeseries=True,
                 recompile_h5_from_csv=False,
                 preload_ram=False):
        """
        :param region: dataset region. choose from "frh01", "frh02", "frh03", "frh04", "belle-ile"
        :param root: where the data will be stored. defaults to `./breizhcrops_dataset`
        :param year: year of the data. currently only `2017`
        :param level: Sentinel 2 processing level. Either `L1C` (top of atmosphere) or `L2A` (bottom of atmosphere)
        :param transform: a transformation function applied to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton
        :param target_transform: a transformation function applied to the label.
        :param filter_length: time series shorter than `filter_length` will be ignored
        :param bool verbose: verbosity flag
        :param bool load_timeseries: if False, no time series data will be loaded. Only index file and class initialization. Used mostly for tests
        :param bool recompile_h5_from_csv: downloads raw csv files and recompiles the h5 databases. Only required when dealing with new datasets
        :param bool preload_ram: loads all time series data in RAM at initialization. Can speed up training if data is stored on HDD.
        """
        assert year in [2017]
        assert level in ["L1C", "L2A"]
        assert region in ["frh01", "frh02", "frh03", "frh04", "belle-ile"]

        if transform is None:
            transform = get_default_transform(level)
        if target_transform is None:
            target_transform = get_default_target_transform()
        self.transform = transform
        self.target_transform = target_transform

        self.region = region.lower()
        self.bands = BANDS[level]

        self.verbose = verbose
        self.year = year
        self.level = level

        if verbose:
            print(f"Initializing BreizhCrops region {region}, year {year}, level {level}")

        self.root = root
        self.h5path, self.indexfile, self.codesfile, self.shapefile, self.classmapping, self.csvfolder = \
            self.build_folder_structure(self.root, self.year, self.level, self.region)

        self.load_classmapping(self.classmapping)

        if not os.path.exists(self.indexfile):
            download_file(INDEX_FILE_URLs[year][level][region], self.indexfile)

        self.index = pd.read_csv(self.indexfile, index_col=0)
        if verbose:
            print(f"loaded {len(self.index)} time series references from {self.indexfile}")

        if load_timeseries and ((not os.path.exists(self.h5path))
                                or (not os.path.getsize(self.h5path) == FILESIZES[year][level][region])):
            if recompile_h5_from_csv:
                self.download_csv_files()
                self.write_h5_database_from_csv()
            else:
                self.download_h5_database()

        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")

        # filter zero-length time series
        self.index = self.index.loc[self.index.sequencelength > filter_length].set_index("idx")

        self.maxseqlength = int(self.index["sequencelength"].max())

        if not os.path.exists(self.codesfile):
            download_file(CODESURL, self.codesfile)
        self.codes = pd.read_csv(self.codesfile, delimiter=";", index_col=0)

        if preload_ram:
            self.X_list = list()
            with h5py.File(self.h5path, "r") as dataset:
                for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
                    self.X_list.append(np.array(dataset[(row.path)]))
        else:
            self.X_list = None

        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)

        #if "id" not in self.index.columns:
            # parse field id from csv path
        self.index["id"] = self.index["path"].apply(lambda path: int(os.path.splitext(os.path.basename(path))[0]))

        # drop fields that are not in the class mapping
        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        self.index[["classid", "classname"]] = self.index["CODE_CULTU"].apply(lambda code: self.mapping.loc[code])
        self.index["region"] = self.region

        self.get_codes()

    def download_csv_files(self):
        zipped_file = os.path.join(self.root, str(self.year), self.level, f"{self.region}.zip")
        download_file(RAW_CSV_URL[self.year][self.level][self.region], zipped_file)
        unzip(zipped_file, self.csvfolder)
        os.remove(zipped_file)

    def build_folder_structure(self, root, year, level, region):
        """
        folder structure

        <root>
           codes.csv
           classmapping.csv
           <year>
              <region>.shp
              <level>
                 <region>.csv
                 <region>.h5
                 <region>
                     <csv>
                         123123.csv
                         123125.csv
                         ...
        """
        year = str(year)

        os.makedirs(os.path.join(root, year, level, region), exist_ok=True)

        h5path = os.path.join(root, year, level, f"{region}.h5")
        indexfile = os.path.join(root, year, level, region + ".csv")
        codesfile = os.path.join(root, "codes.csv")
        shapefile = os.path.join(root, year, f"{region}.shp")
        classmapping = os.path.join(root, "classmapping.csv")
        csvfolder = os.path.join(root, year, level, region, "csv")

        return h5path, indexfile, codesfile, shapefile, classmapping, csvfolder

    def get_fid(self, idx):
        return self.index[self.index["idx"] == idx].index[0]

    def download_h5_database(self):
        print(f"downloading {self.h5path}.tar.gz")
        download_file(H5_URLs[self.year][self.level][self.region], self.h5path + ".tar.gz", overwrite=True)
        print(f"extracting {self.h5path}.tar.gz to {self.h5path}")
        untar(self.h5path + ".tar.gz")
        print(f"removing {self.h5path}.tar.gz")
        os.remove(self.h5path + ".tar.gz")
        print(f"checking integrity by file size...")
        assert os.path.getsize(self.h5path) == FILESIZES[self.year][self.level][self.region]
        print("ok!")

    def write_h5_database_from_csv(self):
        with h5py.File(self.h5path, "w") as dataset:
            for idx, row in tqdm(self.index.iterrows(), total=len(self.index), desc=f"writing {self.h5path}"):
                X = self.load(os.path.join(self.root, row.path))
                dataset.create_dataset(row.path, data=X)

    def get_codes(self):
        return self.codes

    def geodataframe(self):

        if not os.path.exists(self.shapefile):
            targzfile = os.path.join(os.path.dirname(self.shapefile), self.region + ".tar.gz")
            download_file(SHP_URLs[self.year][self.region], targzfile)
            untar(targzfile)
            os.remove(targzfile)


        geodataframe = gpd.GeoDataFrame(self.index.set_index("id"))

        # copy geometry from shapefile to index file
        geom = gpd.read_file(self.shapefile).set_index("ID")
        geom.index.name = "id"
        geodataframe["geometry"] = geom["geometry"]
        geodataframe.crs = geom.crs

        return geodataframe.reset_index()

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            if self.verbose:
                print(f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}")
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        if self.verbose:
            print(f"read {self.nclasses} classes from {classmapping}")

    def load(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        sample = pd.read_csv(os.path.join(self.csvfolder, os.path.basename(csv_file)), index_col=0).dropna()
        # convert datetime to int
        sample["doa"] = pd.to_datetime(sample["doa"]).astype(int)
        sample = sample.groupby(by="doa").first().reset_index()
        X = np.array(sample[self.bands].values)

        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0
            X = X[~t_without_nans]

        return X

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]

        if self.X_list is None:
            with h5py.File(self.h5path, "r") as dataset:
                X = np.array(dataset[(row.path)])
        else:
            X = self.X_list[index]

        # translate CODE_CULTU to class id
        y = self.mapping.loc[row["CODE_CULTU"]].id

        #npad = self.maxseqlength - X.shape[0]
        #X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=PADDING_VALUE)

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y, row.id

def get_default_transform(level):

    #padded_value = PADDING_VALUE
    sequencelength = 45

    bands = BANDS[level]
    if level == "L1C":
        selected_bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
    elif level == "L2A":
        selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

    def transform(x):
        #x = x[x[:, 0] != padded_value, :]  # remove padded values

        # choose selected bands
        x = x[:, selected_band_idxs] * 1e-4  # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor)
    return transform

def get_default_target_transform():
    return lambda y: torch.tensor(y, dtype=torch.long)


if __name__ == '__main__':
    BreizhCrops(region="frh03", root="/tmp", load_timeseries=False, level="L2A",recompile_h5_from_csv=True)
