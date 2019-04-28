#!/usr/bin/env bash

for region in "frh01" "frh02" "frh03" "frh04"; do
python src/query_gee.py data/shp/raw/${region^^}.shp --start 2017-01-01 --end 2017-12-31 --label-col CODE_CULTU --id-col ID --outfolder data/csv/$region
done