import os
import geopandas as gpd
import pandas as pd

annotated_shape_file_folder = "/data/france/BreizhCrops/shp/annotated"
classmapping_csv = "/data/france/BreizhCrops/classmapping.csv"

minimum_instances_per_region = 250
minimum_instances_total = 1000

regions = dict(
    frh01=gpd.read_file(os.path.join(annotated_shape_file_folder, "frh01.shp")),
    frh02=gpd.read_file(os.path.join(annotated_shape_file_folder, "frh02.shp")),
    frh03=gpd.read_file(os.path.join(annotated_shape_file_folder, "frh03.shp")),
    frh04=gpd.read_file(os.path.join(annotated_shape_file_folder, "frh04.shp"))
)

counts = list()
for name, data in regions.items():
    count = data.groupby("group_name").count()["ID"]
    count.name=name
    counts.append(count)
counts = gpd.GeoDataFrame(counts).fillna(0).astype(int).T

# select only groups that are somewhat common in the dataset
counts = counts.loc[(counts >= minimum_instances_per_region).all(1) & (counts.sum(1) >= minimum_instances_total)]
counts["group_name"]=counts.index
group_table = counts.reset_index(drop=True)
group_table["id"] = group_table.index

codes_all = pd.concat([region[["CODE_CULTU","group_name"]] for name, region in regions.items()],axis=0)
codes = codes_all.groupby("CODE_CULTU").first()

# filter code groups with
codes = codes.loc[codes["group_name"].isin(counts.index)]
codes = codes.reset_index()

classmapping = pd.merge(group_table[["id","group_name"]], codes, on="group_name")
classmapping = classmapping.rename(columns={'group_name': 'classname', 'CODE_CULTU': 'code'})

print("writing "+classmapping_csv)
classmapping.to_csv(classmapping_csv)
