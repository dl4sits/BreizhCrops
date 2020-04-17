#!/usr/bin/env bash

for region in "frh01" "frh02" "frh03" "frh04"; do
python breizhcrops/processing/query_gee.py /data2/france2018/${region}.shp --start 2018-01-01 --end 2018-12-31 --label-col CODE_CULTU --id-col ID_PARCEL --outfolder /data2/france2018/csv/$region
done