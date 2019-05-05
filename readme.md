# BreizhCrops

time series example of meadows

<img src=doc/exampletop.png>

Time Series example of corn

<img src=doc/examplebottom.png>

field parcels and administrative NUTS-2 regions

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

baseline models located in `models`

* `LSTM` located in `src/rnn.py`
* `transformer-encoder` adopted `https://github.com/jadore801120/attention-is-all-you-need-pytorch`
