import geopandas as gpd
import os

def main():
    shpfolder = "/data/france/BreizhCrops/shp/raw/"
    csvfolder = "/data/france/BreizhCrops/csv/FRH0"
    idsfolder = "/data/france/BreizhCrops/ids"

    regions = dict(
        frh01=gpd.read_file(os.path.join(shpfolder, "FRH01.shp")),
        frh02=gpd.read_file(os.path.join(shpfolder, "FRH02.shp")),
        frh03=gpd.read_file(os.path.join(shpfolder, "FRH03.shp")),
        frh04=gpd.read_file(os.path.join(shpfolder, "FRH04.shp"))
    )

    exists = dict()
    not_exists = dict()
    for name, data in regions.items():
        exists[name], not_exists[name] = check_exists(data.ID, csvfolder)

        print("Region {}: found {} samples. {} missing".format(name, len(exists[name]), len(not_exists[name])))

    for name, ids in exists.items():
        outfile = os.path.join(idsfolder, name + ".txt")
        print("writing " + outfile)
        with open(outfile, 'w') as f:
            for item in ids:
                f.write("%s\n" % item)

def check_exists(ids, csvfolder):
    """
    iterates through the polygon ids and checks if {id}.csv exists in csvfolder
    returns list of polygons that exists and not_exist in the folder
    """

    not_exists = list()
    exists = list()

    for parcel_id in ids:

        if os.path.exists(os.path.join(csvfolder, str(parcel_id) + ".csv")):
            exists.append(parcel_id)
        else:
            not_exists.append(parcel_id)

    return exists, not_exists

if __name__=="__main__":
    main()