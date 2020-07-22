import os
import geopandas as gpd
import pandas as pd

from ..breizhcrops.datasets.urls import RAW_CSV_URL
from ..breizhcrops.utils import download_file, unzip


def main(self, region, root="breizhcrops_dataset", year=2018, level="L1C"):

    dowload_csv_files(self) # ??

    path = self.csvfolder
    dir_list = os.listdir(path)

    for i in range(0, len(dir_list)):
        file = load(dir_list[i])
        df = pd.read_csv(file)
        


        sequencelength = len(df)

    # id = id
    # CODE_CULTU = label
    # path = csv/frh0x
    # meanQA60 =
    sequencelength = len(df)



    write_indexfile()


def download_csv_files(self):
    zipped_file = os.path.join(self.root, str(self.year), self.level, f"{self.region}.zip")
    download_file(RAW_CSV_URL[self.year][self.level][self.region], zipped_file)
    unzip(zipped_file, self.csvfolder)
    os.remove(zipped_file)

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


def write_indexfile(indexfile):

    print("writing " + index_file_csv)
    indexfile.to_csv(index_file_csv)