import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import geopandas as gpd

from ..utils import download_file

FRH01URL = "https://syncandshare.lrz.de/dl/fiA33ywfHQdzbxXwYQ5zLVpp/frh01.zip"
FRH02URL = "https://syncandshare.lrz.de/dl/fi2pg7sXMjTQRSzrWxRdGLux/frh02.zip"
FRH03URL = "https://syncandshare.lrz.de/dl/fiFbj4sqWYzd4kmcEThTJzYC/frh03.zip"
FRH04URL = "https://syncandshare.lrz.de/dl/fi6rE9rgVyPqS5T5yN6Fccv5/frh04.zip"

FRH01_IDS_URL = "https://syncandshare.lrz.de/dl/fiHbqfHQpyf2UnsoJw1HRYpD/frh01.txt"
FRH02_IDS_URL = "https://syncandshare.lrz.de/dl/fiTeaWJTvL62dQbLxrjH9kmM/frh02.txt"
FRH03_IDS_URL = "https://syncandshare.lrz.de/dl/fiUE2J3DW6MR6NNRijaswMuS/frh03.txt"
FRH04_IDS_URL = "https://syncandshare.lrz.de/dl/fiBUWMD2TTRsHcLzmKYnkFPi/frh04.txt"

FRH01_IDX_URL = "https://syncandshare.lrz.de/dl/fiE7ExSPEF5j1LHADGZ1GcAV/frh01.csv"
FRH02_IDX_URL = "https://syncandshare.lrz.de/dl/fiEutoBWs3JFjCfJpoLWq5HF/frh02.csv"
FRH03_IDX_URL = "https://syncandshare.lrz.de/dl/fiJL3LMrzYwmULbvzFiyVZuY/frh03.csv"
FRH04_IDX_URL = "https://syncandshare.lrz.de/dl/fiCntufUMakKdjWZNq8eS5vw/frh04.csv"

FRH01_SHP_URL = "https://syncandshare.lrz.de/dl/fiAHe7ZYMerBi2yJ5hKJmTXS/frh01.tar.gz"
FRH02_SHP_URL = "https://syncandshare.lrz.de/dl/fi8L5iwpJXs2b9hKEFjQoML5/frh02.tar.gz"
FRH03_SHP_URL = "https://syncandshare.lrz.de/dl/fiTdWAa8b9K4XVmrBbZ6413i/frh03.tar.gz"
FRH04_SHP_URL = "https://syncandshare.lrz.de/dl/fiKfoL1VW9jiDXPgnVXu7ZFK/frh04.tar.gz"

FRH01_NPZ_URL = "https://syncandshare.lrz.de/dl/fiAMZX6kN6SfwEJKmznqtgAd/frh01.npz"
FRH02_NPZ_URL = "https://syncandshare.lrz.de/dl/fi8LqK94Kew7fzBkG5SPQ3Cx/frh02.npz"
FRH03_NPZ_URL = "https://syncandshare.lrz.de/dl/fiTMHJAKvU8b5PazC1HJSuF9/frh03.npz"
FRH04_NPZ_URL = "https://syncandshare.lrz.de/dl/fi337Bzz6xbUGRM9AwT7q1up/frh04.npz"

CLASSMAPPINGURL = "https://syncandshare.lrz.de/dl/fiWcv23b3PxswYZFh2htEpSs/classmapping.csv"
CODESURL = "https://syncandshare.lrz.de/dl/fiFVnHYsEsix7HTGYRh6Zh3/codes.csv"

class BreizhCrops(Dataset):

    def __init__(self, region, root="data",
                 classmapping=None,
                 transform = None, target_transform = None, padding_value=-1,
                 filter_length=0, verbose=False, load_timeseries=True):
        self.region = region.lower()
        print("Initializing BreizhCrops region {}".format(self.region))

        self.bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa']

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.padding_value = padding_value
        self.verbose=verbose

        self.load_classmapping(classmapping)

        indexfile = os.path.join(self.root, region + ".csv")
        if not os.path.exists(indexfile):
            download_file(get_idx_url(region),indexfile)

        index = pd.read_csv(indexfile, index_col=0)
        if verbose:
            print(f"loaded {len(index)} time series references from {indexfile}")

        self.index = index.loc[index["CODE_CULTU"].isin(self.mapping.index)]
        if verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")

        # filter zero-length time series
        self.index = self.index.loc[self.index.sequencelength > filter_length]

        if load_timeseries:
            self.load_timeseries()

        self.get_codes()

    def load_timeseries(self):
        cachefile = os.path.join(self.root,f"{self.region}.npz")

        if not os.path.exists(cachefile):
            download_file(get_npz_url(self.region),cachefile)

        if self.verbose:
            print(f"loading data from {cachefile}")
        self.X, self.y, self.id = self.load_cache(cachefile)

        self.maxseqlength = self.index["sequencelength"].max()
        self.ids = self.index.idx.values

    def get_fid(self,idx):
        return self.index[self.index["idx"] == idx].index[0]

    def get_codes(self):
        codesfile = os.path.join(self.root,"codes.csv")
        if not os.path.exists(codesfile):
            download_file(CODESURL, codesfile)
        return pd.read_csv(codesfile,delimiter=";",index_col=0)

    def geodataframe(self):
        shapefile = os.path.join(self.root,"shp",f"{self.region}.shp")

        if not os.path.exists(shapefile):
            targzfile = os.path.join(os.path.dirname(shapefile),self.region+".tar,gz")
            download_file(get_shp_url(self.region), targzfile)
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

    def load_cache(self, cachefile):
        with np.load(cachefile,allow_pickle=True) as f:
            X = f["X"]
            y = f["y"]
            id = f["id"]
        return X, y, id

    def cache(self):
        import geopandas as gpd
        if self.verbose:
            print("loading shapefile {}")
        shapefile = os.path.join(self.root,"shp",self.region.upper()+".shp")
        index = gpd.read_file(shapefile)[["ID", "CODE_CULTU"]].set_index("ID")

        #index = pd.read_csv(os.path.join(root,region+".csv"), index_col=0)
        index.index.name = "id"
        csvfiles = [(int(os.path.splitext(csv)[0]), os.path.join("csv", self.region, csv)) for csv in
                    os.listdir(self.data_folder) if csv.endswith(".csv")]
        csvfiles = pd.DataFrame(csvfiles, columns=["id", "path"]).set_index("id")
        index = pd.concat([index, csvfiles], axis=1, join="inner")

        X_list = list()
        stats = list()
        sample_index=0
        with tqdm(index.iterrows(), total=len(index)) as iterator:
            for idx, row in iterator:
                try:
                    X = self.load(os.path.join(self.root, row.path))
                    X_list.append(X)
                    stats.append(
                        dict(
                            id=idx,
                            sequencelength=X.shape[0],
                            meanQA60=X[:,self.bands.index("QA60")].mean(),
                            idx=sample_index
                        )
                    )
                    sample_index += 1
                except:
                    print(f"Could not load {row.path}. skipping...")

        stats = pd.DataFrame(stats).set_index("id")

        index = pd.concat([index, stats], axis=1, join="inner")

        fn = os.path.join(self.root,self.region+"_complete.csv")
        print(f"saving {fn}")
        index.to_csv(fn)

        fn = os.path.join(self.root,f"{self.region}.npz")
        print(f"saving {fn}")
        np.savez_compressed(fn, X=np.array(X_list), y=index["CODE_CULTU"].values, id=index.index.values)

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

    def applyclassmapping(self, nutzcodes):
        """uses a mapping table to replace nutzcodes (e.g. 451, 411) with class ids"""
        return np.array([self.mapping.loc[nutzcode]["id"] for nutzcode in nutzcodes])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        X = self.X[id]
        y = self.y[id]

        npad = self.maxseqlength - X.shape[0]
        X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=self.padding_value)

        if self.transform is not None:
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y

def get_idx_url(region):
    if region == "frh01":
        url = FRH01_IDX_URL
    elif region == "frh02":
        url = FRH02_IDX_URL
    elif region == "frh03":
        url = FRH03_IDX_URL
    elif region == "frh04":
        url = FRH04_IDX_URL
    else:
        raise ValueError(f"region {region} not in ['frh01','frh02','frh03','frh03']")
    return url

def get_shp_url(region):
    if region == "frh01":
        url = FRH01_SHP_URL
    elif region == "frh02":
        url = FRH02_SHP_URL
    elif region == "frh03":
        url = FRH03_SHP_URL
    elif region == "frh04":
        url = FRH04_SHP_URL
    else:
        raise ValueError(f"region {region} not in ['frh01','frh02','frh03','frh03']")
    return url

def get_npz_url(region):
    if region == "frh01":
        url = FRH01_NPZ_URL
    elif region == "frh02":
        url = FRH02_NPZ_URL
    elif region == "frh03":
        url = FRH03_NPZ_URL
    elif region == "frh04":
        url = FRH04_NPZ_URL
    else:
        raise ValueError(f"region {region} not in ['frh01','frh02','frh03','frh03']")
    return url

def get_url(region):
    if region == "frh01":
        url = FRH01URL
    elif region == "frh02":
        url = FRH02URL
    elif region == "frh03":
        url = FRH03URL
    elif region == "frh04":
        url = FRH04URL
    else:
        raise ValueError(f"region {region} not in ['frh01','frh02','frh03','frh03']")
    return url