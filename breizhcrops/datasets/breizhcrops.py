import torch
import torch.utils.data
import pandas as pd
import os
import sys
import numpy as np
from numpy import genfromtxt
from tqdm import tqdm
import urllib

BANDS = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9']

#B1,B10,B11,B12,B2,B3,B4,B5,B6,B7,B8,B8A,B9,QA10,QA20,QA60,doa,label,id

NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

class BreizhCrops(torch.utils.data.Dataset):

    def __init__(self, root, region="FRH01", samplet=70, classmapping=None, cache=True):
        """

        :param root:
        :param region: csv/<regionA>/<id>.csv
        :param partition: one of train/valid/eval
        :param nsamples: load less samples for debug
        :param partitioning_scheme either blocks or random
        """

        # ensure that different seeds are set per partition
        #seed = sum([ord(ch) for ch in partition])
        np.random.seed(0)
        torch.random.manual_seed(0)

        self.root = root

        if classmapping is None:
            classmapping = self.root + "/classmapping.csv"

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname=self.classname
        self.nclasses = len(self.classes)

        self.region = region
        self.data_folder = "{root}/csv/{region}".format(root=self.root, region=self.region)
        self.samplet = samplet

        #all_csv_files
        #self.csvfiles = [ for f in os.listdir(root)]
        print("Initializing CropsDataset {}".format(self.region))

        self.cache = os.path.join(self.root,"npy",region)

        print("read {} classes".format(self.nclasses))

        if cache and self.cache_exists() and not self.mapping_consistent_with_cache():
            self.clean_cache()

        if cache and self.cache_exists() and self.mapping_consistent_with_cache():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through csv folders in " + str(self.data_folder))
            self.cache_dataset()

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)

        print("loaded {} samples".format(len(self.ids)))
        #print("class frequencies " + ", ".join(["{c}:{h}".format(h=h, c=c) for h, c in zip(self.hist, self.classes)]))

    def read_ids(self):

        ids_file = os.path.join(self.root, "ids", self.region + ".txt")
        with open(ids_file, "r") as f:
            ids = [int(id) for id in f.readlines()]

        return ids

    def cache_dataset(self):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        #ids = self.split(self.partition)
        ids = self.read_ids()

        self.X = list()
        self.nutzcodes = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        self.samples = list()
        i = 0
        for id in ids:
            if i%500==0:
                update_progress(i/float(len(ids)))
            i+=1

            id_file = self.data_folder+"/{id}.csv".format(id=id)
            if os.path.exists(id_file):
                self.samples.append(id_file)

                X,nutzcode = self.load(id_file)

                if len(nutzcode) > 0:

                    # only take first since class id does not change through time
                    nutzcode = nutzcode[0]

                    # drop samples where nutzcode is not in mapping table
                    if nutzcode in self.mapping.index:

                        # replace nutzcode with class id- e.g. 451 -> 0, 999 -> 1
                        #y = self.mapping.loc[y]["id"]

                        self.X.append(X)
                        self.nutzcodes.append(nutzcode)
                        self.ids.append(id)

            else:
                self.stats["not_found"].append(id_file)

        self.y = self.applyclassmapping(self.nutzcodes)

        self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
        self.sequencelength = self.sequencelengths.max()
        self.ndims = np.array(X).shape[1]

        self.hist,_ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist
        #if 0 in self.hist:
        #    classid_ = np.argmin(self.hist)
        #    nutzid_ = self.mapping.iloc[classid_].name
        #    raise ValueError("Class {id} (nutzcode {nutzcode}) has 0 occurences in the dataset! "
        #                     "Check dataset or mapping table".format(id=classid_, nutzcode=nutzid_))


        #self.dataweights = np.array([self.classweights[y] for y in self.y])

        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X, self.classweights)

    def mapping_consistent_with_cache(self):
        # cached y must have the same number of classes than the mapping
        return True
        #return len(np.unique(np.load(os.path.join(self.cache, "y.npy")))) == self.nclasses

    def cache_variables(self, y, sequencelengths, ids, ndims, X, classweights):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        #np.save(os.path.join(self.cache, "dataweights.npy"), dataweights)
        np.save(os.path.join(self.cache, "X.npy"), X)

    def load_cached_dataset(self):
        # load
        self.classweights = np.load(os.path.join(self.cache, "classweights.npy"), allow_pickle=True)
        self.y = np.load(os.path.join(self.cache, "y.npy"), allow_pickle=True)
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy"), allow_pickle=True))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"), allow_pickle=True)
        self.sequencelength = self.sequencelengths.max()
        self.ids = np.load(os.path.join(self.cache, "ids.npy"), allow_pickle=True)
        #self.dataweights = np.load(os.path.join(self.cache, "dataweights.npy"))
        self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        #dataweightsexist = os.path.exists(os.path.join(self.cache, "dataweights.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.npy"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        #os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.npy"))
        os.removedirs(self.cache)

    def load(self, csv_file, load_pandas = True):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        if load_pandas:
            sample = pd.read_csv(csv_file, index_col=0).dropna()
            X = np.array((sample[BANDS] * NORMALIZING_FACTOR).values)
            nutzcodes = sample["label"].values
            # nutzcode to classids (451,411) -> (0,1)

        else: # load with numpy
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
            X,nutzcodes = self.load(csvfile)
            y = self.applyclassmapping(nutzcodes=nutzcodes)
        else:

            X = self.X[idx]
            y = np.array([self.y[idx]] * X.shape[0]) # repeat y for each entry in x

        # pad up to maximum sequence length
        t = X.shape[0]

        if self.samplet is None:
            npad = self.sequencelengths.max() - t
            X = np.pad(X,[(0,npad), (0,0)],'constant', constant_values=PADDING_VALUE)
            y = np.pad(y, (0, npad), 'constant', constant_values=PADDING_VALUE)
        else:
            idxs = np.random.choice(t, self.samplet, replace=False)
            idxs.sort()
            X = X[idxs]
            y = y[idxs]


        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rLoaded: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

if __name__=="__main__":
    root = "/home/marc/projects/BreizhCrops/data"
    CropsDataset(root=root, region="frh01")
    CropsDataset(root=root, region="frh02")
    CropsDataset(root=root, region="frh03")
    CropsDataset(root=root, region="frh04")
