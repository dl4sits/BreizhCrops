# BreizhCrops:
#### A Satellite Time Series Dataset for Crop Type Identification

![](https://github.com/tum-lmf/breizhcrops/workflows/build-package/badge.svg)


[Paper](https://arxiv.org/abs/1905.11893) presented at the [ICML 2019 Time Series workshop, Long Beach, USA](http://roseyu.com/time-series-workshop/)
```
@article{russwurm2019:BreizhCrops,
  author    = {Marc Ru{\ss}wurm and
               S{\'{e}}bastien Lef{\`{e}}vre and
               Marco K{\"{o}}rner},
  title     = {BreizhCrops: A Satellite Time Series Dataset for Crop Type Identification},
  journal   = {CoRR},
  volume    = {abs/1905.11893},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.11893},
  archivePrefix = {arXiv},
  eprint    = {1905.11893},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-11893},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

<a href=https://arxiv.org/abs/1905.11893><img height=300px src=doc/paper.png /></a>
<a href="doc/poster.pdf"><img height=300px src=doc/poster.png /></a>

### Installation

#### Anaconda Environment Setup

```
conda create -n breizhcrops python=3.8 pip
conda activate breizhcrops 
```

intall from GitHub
```
pip install git+https://github.com/tum-lmf/breizhcrops
```

install from sources
```
git clone https://github.com/tum-lmf/breizhcrops
cd BreizhCrops
pip install .
```

### Usage

```
from breizhcrops import BreizhCrops
from breizhcrops.models import LSTM, TransformerEncoder, TempCNN, MSResNet
```

### Download Dataset

Simply initialize the Dataset

```
BreizhCrops(root="data",region="frh01")
```

## Organization in NUTS Administrative Regions

<img width=54% src=doc/BrittanyParcels.png>
<img width=45% src=doc/regions.png>


### Data organization

```
# mapping from ~160 categories to 13 most frequenc groups
data/classmapping.csv

# csv files
data/csv/frh0{1,2,3,4}/*.csv

# polygon ids per departement
data/ids/frh0{1,2,3,4}.txt

# cached numpy arrays for faster data loading
data/frh0{1,2,3,4}/*.npz

# raw shapefile geometries with labels
data/shp/*
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Data organization scripts

#### Query satellite data from Google Earth Engine

```
python src/query_gee.py data/shp/raw/frh01.shp --start 2017-01-01 --end 2017-12-31 --label-col CODE_CULTU --id-col ID --outfolder data/csv/frh01
```
