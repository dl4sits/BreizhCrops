import ee
import pandas as pd
import geojson
import datetime
import geopandas as gpd
import os
import argparse
import time
from tqdm import tqdm
import urllib3

def parse_args():
    parser = argparse.ArgumentParser(description='Query Google Earth Engine for reflectance data from'
                                                 'Satellites.'
                                                 'Data is queried from in a certain period over features'
                                                 'of a provided shapefile')
    parser.add_argument('shapefile', type=str,
                        help="path to shapefile containing attributes 'id', 'label, 'class''")
    parser.add_argument('--start', type=str, help='start date YY-MM-DD')
    parser.add_argument('--end', type=str, help='end date YY-MM-DD')
    parser.add_argument('--outfolder', type=str, default="/tmp", help='output directory')
    parser.add_argument('--scale', type=int, default=30, help='scale of pixels at query time')
    parser.add_argument('--aggregate-method', type=int, default=30, help='scale of pixels at query time')
    parser.add_argument('--id-col', type=str, default="id", help='id column name in shapefile')
    parser.add_argument('--label-col', type=str, default="label", help='label column name in shapefile')
    parser.add_argument('--collection', type=str, default="COPERNICUS/S2",
                        help="Google Earth Engine collection ID. E.g. 'COPERNICUS/S2'")
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    return parser.parse_args()

def main(start, end, shapefile, outfolder, scale, collection, label_col, id_col):

    # load and reproject to lat lon
    df = gpd.read_file(shapefile).to_crs({'init': 'epsg:4326'})
    print("read {} geometries from {}".format(len(df), shapefile))

    with tqdm(df.T.iteritems(), total=len(df)) as pbar:
        for iter, row in pbar:

            try:
                outfile = os.path.join(outfolder, "{}.csv".format(row[id_col]))

                if os.path.exists(outfile):
                    print("file {} exists. skipping...".format(outfile))
                    continue

                region = shapely2ee(row["geometry"])

                getinfo_dict = query(region, start, end, scale, collection)
                dataframe = parse(getinfo_dict)

                dataframe["label"] = row[label_col]
                dataframe["id"] = row[id_col]

                now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                pbar.set_description_str(str(now) +": writing " + outfile)
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                dataframe.to_csv(outfile)
            except ee.ee_exception.EEException as e:
                print(e)
                print("Skipping in 2 seconds")
                time.sleep(2)
            except AttributeError as e:
                print(e)
                print("geometry id {} invalid. ".format(row[id_col]))
                #time.sleep(2)
            except urllib3.exceptions.ProtocolError as e:
                print(e)
                print("Connection aborted. geometry id {}. Skipping in 2 seconds".format(row[id_col]))
                time.sleep(2)


def shapely2ee(geometry):
    pt_list = list(zip(*geometry.exterior.coords.xy))
    return ee.Geometry.Polygon(pt_list)

def load_geojson(file):
    with open(file) as f:
        gj = geojson.load(f)
    pt_list = gj['features'][0]["geometry"]["coordinates"][0]
    return ee.Geometry.Polygon(pt_list)

def query(region, start, end, scale, collection):
    images = ee.ImageCollection(collection).filterDate(start, end).filterBounds(region)

    def _reduce_region(image):
        stat_dict = image.reduceRegion(ee.Reducer.mean(), region, scale);
        return ee.Feature(None, stat_dict)

    return images.map(_reduce_region).getInfo()

def parse(query_dict):
    ids = [feature["id"] for feature in query_dict["features"]]

    properties = [feature["properties"] for feature in query_dict["features"]]
    df = pd.DataFrame(properties, index=ids)

    # parse date of acquisition from the id
    df["doa"] = [datetime.datetime.strptime(id[0:8], "%Y%m%d").strftime("%Y-%m-%d") for id in ids]

    return df

if __name__=="__main__":
    ee.Initialize()

    args = parse_args()

    main(args.start,args.end,args.shapefile,args.outfolder,args.scale, args.collection, args.label_col, args.id_col)
