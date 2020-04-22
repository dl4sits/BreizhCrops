import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import geopandas as gpd
import h5py
import tarfile

from ..utils import download_file

RAW_CSV_URL = dict(
    frh01="https://syncandshare.lrz.de/dl/fiA33ywfHQdzbxXwYQ5zLVpp/frh01.zip",
    frh02="https://syncandshare.lrz.de/dl/fi2pg7sXMjTQRSzrWxRdGLux/frh02.zip",
    frh03="https://syncandshare.lrz.de/dl/fiFbj4sqWYzd4kmcEThTJzYC/frh03.zip",
    frh04="https://syncandshare.lrz.de/dl/fi6rE9rgVyPqS5T5yN6Fccv5/frh04.zip",
)

INDEX_FILE_URLs = dict(
    frh01="https://syncandshare.lrz.de/dl/fiE7ExSPEF5j1LHADGZ1GcAV/frh01.csv",
    frh02="https://syncandshare.lrz.de/dl/fiEutoBWs3JFjCfJpoLWq5HF/frh02.csv",
    frh03="https://syncandshare.lrz.de/dl/fiJL3LMrzYwmULbvzFiyVZuY/frh03.csv",
    frh04="https://syncandshare.lrz.de/dl/fiCntufUMakKdjWZNq8eS5vw/frh04.csv",
)


SHP_URLs = dict(
    frh01="https://syncandshare.lrz.de/dl/fiAHe7ZYMerBi2yJ5hKJmTXS/frh01.tar.gz",
    frh02="https://syncandshare.lrz.de/dl/fi8L5iwpJXs2b9hKEFjQoML5/frh02.tar.gz",
    frh03="https://syncandshare.lrz.de/dl/fiTdWAa8b9K4XVmrBbZ6413i/frh03.tar.gz",
    frh04="https://syncandshare.lrz.de/dl/fiKfoL1VW9jiDXPgnVXu7ZFK/frh04.tar.gz",
)

FILESIZES = dict(
    frh01=2559635960,
    frh02=2253658856,
    frh03=2493572704,
    frh04=1555075632
)

H5_URLs = dict(
    frh01="https://syncandshare.lrz.de/dl/fiFe2C3qDW5MWnVtWdaAT7xC/frh01.h5.tar.gz",
    frh02="https://syncandshare.lrz.de/dl/fi3dyXpipntJyiCZZJdLNcTi/frh02.h5.tar.gz",
    frh03="https://syncandshare.lrz.de/dl/fi8ahoBEbekCKh61PxDAvjQ/frh03.h5.tar.gz",
    frh04="https://syncandshare.lrz.de/dl/fi77rzsEJMWXumq3jpi1VPYF/frh04.h5.tar.gz"
)

# 9-classes used in ISPRS submission
CLASSMAPPINGURL = "https://syncandshare.lrz.de/dl/fiWcv23b3PxswYZFh2htEpSs/classmapping.csv"

# 13-classes used in ICML workshop
CLASSMAPPINGURL_ICML = "https://syncandshare.lrz.de/dl/fiAXzNVSgAz7sKBdonhsCpkG/classmapping_icml.csv"

CODESURL = "https://syncandshare.lrz.de/dl/fiFVnHYsEsix7HTGYRh6Zh3/codes.csv"

class BreizhCrops(Dataset):

    def __init__(self, region, root="data",
                 classmapping=None,
                 transform = None, target_transform = None, padding_value=-1,
                 filter_length=0, verbose=False, load_timeseries=True, recompile_h5_from_csv=False):
        self.region = region.lower()
        if verbose:
            print("Initializing BreizhCrops region {}".format(self.region))

        self.bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa']

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.padding_value = padding_value
        self.verbose = verbose

        self.load_classmapping(classmapping)

        indexfile = os.path.join(self.root, region + ".csv")
        if not os.path.exists(indexfile):
            download_file(INDEX_FILE_URLs[region],indexfile)

        self.index = pd.read_csv(indexfile, index_col=0)
        if verbose:
            print(f"loaded {len(self.index)} time series references from {indexfile}")

        self.h5path = os.path.join(self.root, f"{self.region}.h5")
        if load_timeseries and ((not os.path.exists(self.h5path))
                or (not os.path.getsize(self.h5path) == FILESIZES[region])):
            if recompile_h5_from_csv:
                self.write_h5_database_from_csv()
            else:
                self.download_h5_database()

        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")

        # filter zero-length time series
        self.index = self.index.loc[self.index.sequencelength > filter_length].set_index("idx")

        self.maxseqlength = self.index["sequencelength"].max()

        codesfile = os.path.join(self.root,"codes.csv")
        if not os.path.exists(codesfile):
            download_file(CODESURL, codesfile)
        self.codes = pd.read_csv(codesfile,delimiter=";",index_col=0)

        self.get_codes()

    def get_fid(self,idx):
        return self.index[self.index["idx"] == idx].index[0]

    def download_h5_database(self):
        print(f"downloading {self.h5path}.tar.gz")
        download_file(H5_URLs[self.region], self.h5path+".tar.gz", overwrite=True)
        print(f"extracting {self.h5path}.tar.gz to {self.h5path}")
        untar(self.h5path + ".tar.gz")
        print(f"removing {self.h5path}.tar.gz")
        os.remove(self.h5path+".tar.gz")
        print(f"checking integrity by file size...")
        assert os.path.getsize(self.h5path) == FILESIZES[self.region]
        print("ok!")

    def write_h5_database_from_csv(self):
        with h5py.File(self.h5path, "w") as dataset:
            for idx, row in tqdm(self.index.iterrows(), total=len(self.index), desc=f"writing {self.h5path}"):
                X = self.load(os.path.join(self.root, row.path))
                dataset.create_dataset(row.path, data=X)

    def get_codes(self):
        return self.codes

    def geodataframe(self):
        shapefile = os.path.join(self.root,"shp",f"{self.region}.shp")

        if not os.path.exists(shapefile):
            targzfile = os.path.join(os.path.dirname(shapefile),self.region+".tar,gz")
            download_file(SHP_URLs[self.region], targzfile)
            import tarfile
            with tarfile.open(targzfile) as tar:
                tar.extractall()

        geom = gpd.read_file(shapefile).set_index("ID")
        geom.index.name = "id"

        geom["sequencelength"] = self.index["sequencelength"]
        geom["meanQA60"] = self.index["meanQA60"]
        geom["cloudCoverage"] = geom["meanQA60"] / 1024  # 1024 indicates complete cloud coverage
        geom["region"] = self.region

        return geom

    def load_classmapping(self,classmapping):
        if classmapping is None:
            classmapping = os.path.join(self.root,"classmapping.csv")
            os.makedirs(self.root, exist_ok=True)
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

        sample = pd.read_csv(csv_file, index_col=0).dropna()
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

        with h5py.File(self.h5path, "r") as dataset:
            X = np.array(dataset[(row.path)])
        y = row["CODE_CULTU"]

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