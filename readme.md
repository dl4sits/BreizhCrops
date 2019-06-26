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

## Data

Time series example of meadows

<img src=doc/exampletop.png>

Time series example of corn

<img src=doc/examplebottom.png>

Field parcels and administrative NUTS-2 regions

<img width=54% src=doc/BrittanyParcels.png>
<img width=45% src=doc/regions.png>

## Download Data

```
wget https://storage.googleapis.com/breizhcrops/data.zip
unzip data.zip
rm data.zip # cleanup
```

## Data organization

```
# mapping from ~160 categories to 13 most frequenc groups
data/classmapping.csv

# csv files
data/csv/frh{1,2,3,4}/*.csv

# polygon ids per departement
data/ids/frh{1,2,3,4}.txt

# cached numpy arrays for faster data loading
data/csv/frh{1,2,3,4}/*.npy

# raw shapefile geometries with labels
data/shp/*
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Data Organizing Steps

```
python write_annotated_shp.py
python write_classmapping.py
python write_tileids.py
```

## Basline Models

Baseline models located in `models`

* `LSTM` located in `src/rnn.py`
* `transformer-encoder` adopted `https://github.com/jadore801120/attention-is-all-you-need-pytorch`
