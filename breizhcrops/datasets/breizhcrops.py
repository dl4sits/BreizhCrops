import os
import tarfile

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from .urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, FILESIZES, SHP_URLs, H5_URLs, RAW_CSV_URL
from ..utils import download_file, unzip

BANDS = {
    "L1C": ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa'],
    "L2A": ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B11', 'B12', 'CLD', 'EDG', 'SAT']
}


class BreizhCrops(Dataset):

    def __init__(self, region, root="data", year=2017, level="L1C",
                 transform=None, target_transform=None, padding_value=-1,
                 filter_length=0, verbose=False, load_timeseries=True, recompile_h5_from_csv=False, preload_ram=False):

        assert year in [2017]
        assert level in ["L1C", "L2A"]
        assert region in ["frh01", "frh02", "frh03", "frh04"]

        self.region = region.lower()
        self.bands = BANDS[level]
        self.transform = transform
        self.target_transform = target_transform
        self.padding_value = padding_value
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

        self.maxseqlength = self.index["sequencelength"].max()

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

        # add classname and id from mapping to the index dataframe
        #mapping = self.mapping.reset_index().rename(columns={"code": "CODE_CULTU"})
        #self.index = self.index.merge(mapping, on="CODE_CULTU")
        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)

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
        import tarfile

        if not os.path.exists(self.shapefile):
            targzfile = os.path.join(os.path.dirname(self.shapefile), self.region + ".tar.gz")
            download_file(SHP_URLs[self.year][self.region], targzfile)
            untar(targzfile)
            os.remove(targzfile)
            #with tarfile.open(targzfile) as tar:
            #    tar.extractall()

        geom = gpd.read_file(self.shapefile).set_index("ID")
        geom.index.name = "id"

        geom["sequencelength"] = self.index["sequencelength"]
        geom["meanCLD"] = self.index["meanCLD"]
        geom["cloudCoverage"] = geom["meanCLD"] / 1024  # 1024 indicates complete cloud coverage
        geom["region"] = self.region

        return geom

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

        npad = self.maxseqlength - X.shape[0]
        X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=self.padding_value)

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y, int(os.path.splitext(os.path.basename(row.path))[0])


def untar(filepath):
    dirname = os.path.dirname(filepath)
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=dirname)

if __name__ == '__main__':
    BreizhCrops(region="frh03", root="/tmp", load_timeseries=False, level="L2A",recompile_h5_from_csv=True)