import multiprocessing as mp
import os
import sys
sys.path.append("../breizhcrops")
import logging
from utils import download_file, unzip, untar
import geopandas as gpd
import numpy as np
import rasterio
import pyproj
import shapely
import rasterio.features
import h5py
from breizhcrops import BreizhCrops
from tqdm import tqdm

bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9', 'QA10', 'QA20', 'QA60']

#client = google.cloud.logging.Client()
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/tmp/shapesplit.log', format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

def pool_init(l):
    """
    multiprocessing init function <- ensures that a global lock is present that all threads can accesss
    :param l:
    :return:
    """
    global lock
    lock = l


def mapping_function(raster, geometry, region, id, h5path, date):
    img, meta, mask = crop_raster_by_geometry(raster, geometry, offset=50)
    storagepath = os.path.join(region, str(id), date)
    with h5py.File(h5path, "a") as dataset:
        dataset.create_dataset(storagepath + "/img", data=img)
        dataset.create_dataset(storagepath + "/mask", data=mask)
        meta["path"] = storagepath

def main():
    data_root = "/data2/breizhcrops/merged"
    region = sys.argv[1]#"frh01"


    h5path = f"/ssd/Breizhcrops/{region}.h5"
    with h5py.File(h5path, "a") as dataset:
        print(f"writing empty dataset {h5path}")

    # sys.argv[1]
    print("initializing breizhcrops object")
    bzh = BreizhCrops(region=region, root="/tmp", level="L1C", load_timeseries=False, recompile_h5_from_csv=False)

    print("loading geodataframe")
    geom = bzh.geodataframe()

    root = os.path.join(data_root, region)

    for rastertif in tqdm(os.listdir(root)):
        date = rastertif.replace(".tif","")

        pool = mp.Pool(mp.cpu_count(), initializer=pool_init, initargs=(mp.Lock(),))

        raster = os.path.join(root, f"{date}.tif")

        with rasterio.open(raster, 'r') as ds:
            crs = dict(ds.crs)
            rasterbounds = ds.bounds

        xmin, ymin, xmax, ymax = rasterbounds
        print(f"reprojecting shapefiles from {geom.crs} to {crs['init']}")
        parcels = geom.to_crs(crs).cx[xmin:xmax, ymin:ymax]

        for idx, row in tqdm(parcels.iterrows(), total=len(parcels), desc=f"writing {h5path}", leave=False):
            pool.apply(mapping_function, args=(raster, row.geometry, row.region, row.id, h5path, date))

        pool.close()
        pool.join()

def crop_raster_by_geometry(raster, geometry, offset=None):
    # expand geometry

    if offset is not None:
        #logger.debug("buffering by " + str(offset))
        bounds = geometry.buffer(offset,
                                 cap_style=shapely.geometry.CAP_STYLE.square,
                                 join_style=shapely.geometry.JOIN_STYLE.bevel).bounds
    else:
        bounds = geometry.bounds

    lock.acquire()
    with rasterio.open(raster, 'r') as ds:
        meta = ds.meta.copy()
        transform = ds.transform

        window = rasterio.windows.from_bounds(*bounds, transform=transform)
        # :12 are the 13 sentinel bands
        img = ds.read(window=window)[:12].astype(int)
    lock.release()
    #sys.exit()

    meta.update({
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, transform)})

    logger.debug("image shape: {}".format(img.shape))

    if img.shape[1] == 0 or img.shape[2] == 0:
        raise ValueError("Image shape invalid: {}".format(img.shape))
    #if np.isnan(img).any():
    #    raise ValueError("Image contains {} nan values".format( np.isnan(img).sum() ))

    mask = rasterio.features.rasterize([(geometry, 1)], out_shape=img.shape[1:3], transform=meta["transform"])

    return img, meta, mask


if __name__ == '__main__':
    main()