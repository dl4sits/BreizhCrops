# BreizhCrops:
#### A Satellite Time Series Dataset for Crop Type Identification

[Paper](https://arxiv.org/abs/1905.11893) presented at the [ICML 2019 Time Series workshop, Long Beach, USA](http://roseyu.com/time-series-workshop/)
```
@InProceedings{russwurm2019breizhcrops,
    title={BreizhCrops: A Satellite Time Series Dataset for Crop Type Identification},
    author={Ru{\ss}wurm, Marc and Lef{\`e}vre, S{\'e}bastien and K{\"o}rner, Marco},
    year={2019},
    booktitle   = {International Conference on Machine Learning (ICML)},
    series      = {Time Series Workshop},
    eprint      = {1905.11893},
    eprintclass = {cs.LG, cs.CV, stat.ML},
    eprinttype  = {arxiv},
}
```

<a href=https://arxiv.org/abs/1905.11893><img height=300px src=doc/paper.png /></a>
<a href="doc/poster.pdf"><img height=300px src=doc/poster.png /></a>

### Notebooks

`TrainEvaluateModels.ipynb` for instructions of model training and inference

`BreizhCrops.ipynb` for additional information on the raw data


### Sentinel 2 Time Series of Field Crop Parcels

Time series example of meadows

<img src=doc/exampletop.png>

Time series example of corn

<img src=doc/examplebottom.png>

## Organization in NUTS Administrative Regions

<img width=54% src=doc/BrittanyParcels.png>
<img width=45% src=doc/regions.png>

### Download Data and Models

Download the data in csv files and cached numpy arrays (~14GB)
```
cd data
bash download.sh
```

Download pre-trained models (22mb)
```
cd models
bash download.sh
```

### Data organization

```
# mapping from ~160 categories to 13 most frequenc groups
data/classmapping.csv

# csv files
data/csv/frh0{1,2,3,4}/*.csv

# polygon ids per departement
data/ids/frh0{1,2,3,4}.txt

# cached numpy arrays for faster data loading
data/csv/frh0{1,2,3,4}/*.npy

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

#### Data Management

```
python write_annotated_shp.py
python write_classmapping.py
python write_tileids.py
```
