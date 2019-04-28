import geopandas as gpd
import pandas as pd
import os

def main():
    codescsv="/home/marc/projects/ICML19_TSW/images/codes.csv"
    shpfolder = "/data/france/BreizhCrops/shp/raw/"
    annotated_shape_file_folder = "/data/france/BreizhCrops/shp/annotated"

    english_group_names = ['common wheat','corn grain and silage','barley','other cereals',
           "rapeseed", "sunflower", "other oilseeds", "protein crops",
           'fibre plants','gel (frozen surfaces without production)','rice',
           'pulses','fodder','estives and heaths',
           "permanent meadows", "temporary meadows", "orchards",
           "vines", "nuts", "olive trees",
           'other industrial crops','vegetables or flowers',
           "sugar cane", "miscellaneous"]

    french_group_names = ['Blé tendre', 'Maïs grain et ensilage', 'Orge', 'Autres céréales',
           'Colza', 'Tournesol', 'Autres oléagineux', 'Protéagineux',
           'Plantes à fibres', 'Gel (surfaces gelées sans production)', 'Riz',
           'Légumineuses à grains', 'Fourrage', 'Estives et landes',
           'Prairies permanentes', 'Prairies temporaires', 'Vergers',
           'Vignes', 'Fruits à coque', 'Oliviers',
           'Autres cultures industrielles', 'Légumes ou fleurs',
           'Canne à sucre', 'Divers']

    codes = pd.read_csv(codescsv, delimiter=";",encoding="utf-8")

    regions = dict(
        frh01=gpd.read_file(os.path.join(shpfolder, "FRH01.shp")),
        frh02=gpd.read_file(os.path.join(shpfolder, "FRH02.shp")),
        frh03=gpd.read_file(os.path.join(shpfolder, "FRH03.shp")),
        frh04=gpd.read_file(os.path.join(shpfolder, "FRH04.shp"))
    )

    mapping = pd.DataFrame([english_group_names,french_group_names],index=["group_name","french_group_names"]).T

    codes = pd.merge(codes,mapping,left_on = "Libellé Groupe Culture",right_on = "french_group_names")

    for name, data in regions.items():
        regions[name] = pd.merge(data,codes,left_on = "CODE_CULTU",right_on = "Code Culture")

    for name, data in regions.items():
        path = annotated_shape_file_folder + "/" + name + ".shp"
        print("saving "+path)
        regions[name].to_file(path, driver='ESRI Shapefile', encoding="utf-8")

if __name__=="__main__":
    main()