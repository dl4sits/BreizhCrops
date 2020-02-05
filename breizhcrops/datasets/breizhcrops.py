import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from numpy import genfromtxt
from tqdm import tqdm

from ..utils import download_file, unzip

BANDS = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
         'B8A', 'B9']

NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

FRH01URL = "https://syncandshare.lrz.de/dl/fiA33ywfHQdzbxXwYQ5zLVpp/frh01.zip"
FRH02URL = "https://syncandshare.lrz.de/dl/fi2pg7sXMjTQRSzrWxRdGLux/frh02.zip"
FRH03URL = "https://syncandshare.lrz.de/dl/fiFbj4sqWYzd4kmcEThTJzYC/frh03.zip"
FRH04URL = "https://syncandshare.lrz.de/dl/fi6rE9rgVyPqS5T5yN6Fccv5/frh04.zip"

FRH01_IDS_URL = "https://syncandshare.lrz.de/dl/fiHbqfHQpyf2UnsoJw1HRYpD/frh01.txt",
FRH02_IDS_URL = "https://syncandshare.lrz.de/dl/fiTeaWJTvL62dQbLxrjH9kmM/frh02.txt",
FRH03_IDS_URL = "https://syncandshare.lrz.de/dl/fiUE2J3DW6MR6NNRijaswMuS/frh03.txt",
FRH04_IDS_URL = "https://syncandshare.lrz.de/dl/fiBUWMD2TTRsHcLzmKYnkFPi/frh04.txt",

CLASSMAPPINGURL = "https://syncandshare.lrz.de/dl/fiWcv23b3PxswYZFh2htEpSs/classmapping.csv"

# expected number of csv files from unzipped dataset files
nsamples_per_region = dict(
    frh01=220987,
    frh02=180325,
    frh03=207854,
    frh04=158338
)


class BreizhCrops(torch.utils.data.Dataset):

    def __init__(self, region, root="data", samplet=70, classmapping=None, cache=True):
        self.region = region.lower()
        print("Initializing CropsDataset {}".format(self.region))

        np.random.seed(0)
        torch.random.manual_seed(0)

        self.root = root

        if classmapping is None:
            classmapping = self.root + "/classmapping.csv"
            if not os.path.exists(classmapping):
                print(f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}")
                download_file(CLASSMAPPINGURL, classmapping)
            else:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        print("read {} classes".format(self.nclasses))

        self.data_folder = "{root}/csv/{region}".format(root=self.root, region=self.region)
        self.samplet = samplet

        if not os.path.exists(self.data_folder):
            print(f"no folder structure {self.data_folder} found. creating...")
            os.makedirs(self.data_folder, exist_ok=True)

        csvfiles = [f for f in os.listdir(self.data_folder) if f.endswith(".csv")]
        if not len(csvfiles) == nsamples_per_region[self.region]:
            print(
                f"found only {len(csvfiles)} of {nsamples_per_region[self.region]} csv files in  {self.data_folder}. downloading...")
            zipfile_path = os.path.join(self.root, self.region + ".zip")
            if not os.path.exists(zipfile_path):
                print(f"downloading zipped dataset to {zipfile_path}")
                download_file(get_url(region), zipfile_path)
            os.makedirs(self.data_folder, exist_ok=True)
            print(f"unzipping {zipfile_path} to {self.data_folder}")
            unzip(zipfile_path, os.path.dirname(self.data_folder))

        self.cache = os.path.join(self.root, "npy", region)

        if cache and self.cache_exists():
            self.clean_cache()

        if cache and self.cache_exists():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print(
                f"no cached dataset found in {self.cache}. iterating through csv folders in {self.data_folder}" + str())
            self.cache_dataset()

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)

        print("loaded {} samples".format(len(self.ids)))
        # print("class frequencies " + ", ".join(["{c}:{h}".format(h=h, c=c) for h, c in zip(self.hist, self.classes)]))

    def read_ids(self):

        ids_root = os.path.join(self.root, "ids")
        os.makedirs(ids_root, exist_ok=True)

        ids_file = os.path.join(ids_root, self.region + ".txt")
        if not os.path.exists(ids_file):
            if self.region == "frh01":
                url = FRH01_IDS_URL
            if self.region == "frh02":
                url = FRH02_IDS_URL
            if self.region == "frh03":
                url = FRH03_IDS_URL
            if self.region == "frh04":
                url = FRH04_IDS_URL

            download_file(url, ids_file)

        with open(ids_file, "r") as f:
            ids = [int(id) for id in f.readlines()]

        return ids

    def cache_dataset(self):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        # ids = self.split(self.partition)
        ids = self.read_ids()

        self.X = list()
        self.nutzcodes = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        self.samples = list()

        pbar = tqdm(ids, total=len(ids))
        for id in pbar:

            id_file = self.data_folder + "/{id}.csv".format(id=id)
            if os.path.exists(id_file):
                self.samples.append(id_file)

                X, nutzcode = self.load(id_file)

                if len(nutzcode) > 0:

                    # only take first since class id does not change through time
                    nutzcode = nutzcode[0]

                    # drop samples where nutzcode is not in mapping table
                    if nutzcode in self.mapping.index:
                        # replace nutzcode with class id- e.g. 451 -> 0, 999 -> 1
                        # y = self.mapping.loc[y]["id"]

                        self.X.append(X)
                        self.nutzcodes.append(nutzcode)
                        self.ids.append(id)

            else:
                self.stats["not_found"].append(id_file)

        self.y = self.applyclassmapping(self.nutzcodes)

        self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
        self.sequencelength = self.sequencelengths.max()
        self.ndims = np.array(X).shape[1]

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist
        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X, self.classweights)

    def cache_variables(self, y, sequencelengths, ids, ndims, X, classweights):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        # np.save(os.path.join(self.cache, "dataweights.npy"), dataweights)
        np.save(os.path.join(self.cache, "X.npy"), X)

    def load_cached_dataset(self):
        # load
        self.classweights = np.load(os.path.join(self.cache, "classweights.npy"), allow_pickle=True)
        self.y = np.load(os.path.join(self.cache, "y.npy"), allow_pickle=True)
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy"), allow_pickle=True))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"), allow_pickle=True)
        self.sequencelength = self.sequencelengths.max()
        self.ids = np.load(os.path.join(self.cache, "ids.npy"), allow_pickle=True)
        # self.dataweights = np.load(os.path.join(self.cache, "dataweights.npy"))
        self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        # dataweightsexist = os.path.exists(os.path.join(self.cache, "dataweights.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.npy"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        # os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.npy"))
        os.removedirs(self.cache)

    def load(self, csv_file, load_pandas=True):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        if load_pandas:
            sample = pd.read_csv(csv_file, index_col=0).dropna()
            X = np.array((sample[BANDS] * NORMALIZING_FACTOR).values)
            nutzcodes = sample["label"].values
            # nutzcode to classids (451,411) -> (0,1)

        else:  # load with numpy
            data = genfromtxt(csv_file, delimiter=',', skip_header=1)
            X = data[:, 1:14] * NORMALIZING_FACTOR
            nutzcodes = data[:, 18]

        # drop times that contain nans
        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0

            X = X[~t_without_nans]
            nutzcodes = nutzcodes[~t_without_nans]

        return X, nutzcodes

    def applyclassmapping(self, nutzcodes):
        """uses a mapping table to replace nutzcodes (e.g. 451, 411) with class ids"""
        return np.array([self.mapping.loc[nutzcode]["id"] for nutzcode in nutzcodes])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        load_file = False
        if load_file:
            id = self.ids[idx]
            csvfile = os.path.join(self.data_folder, "{}.csv".format(id))
            X, nutzcodes = self.load(csvfile)
            y = self.applyclassmapping(nutzcodes=nutzcodes)
        else:

            X = self.X[idx]
            y = np.array([self.y[idx]] * X.shape[0])  # repeat y for each entry in x

        # pad up to maximum sequence length
        t = X.shape[0]

        if self.samplet is None:
            npad = self.sequencelengths.max() - t
            X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=PADDING_VALUE)
            y = np.pad(y, (0, npad), 'constant', constant_values=PADDING_VALUE)
        else:
            idxs = np.random.choice(t, self.samplet, replace=False)
            idxs.sort()
            X = X[idxs]
            y = y[idxs]

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y


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
