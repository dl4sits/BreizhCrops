import sys
import os
this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder,".."))

from breizhcrops import BreizhCrops
import pytest
from itertools import product
from breizhcrops.datasets.breizhcrops import SELECTED_BANDS

TESTS_DATA_ROOT = os.environ.get('TESTS_DATA_ROOT', '/tmp')

""" test if compiling the h5 database from raw CSV files still works.
@pytest.mark.parametrize("region,year,level",
                         product(["frh01", "frh02", "frh03", "frh04"],
                                 [2017, 2018],
                                 ["L1C", "L2A"]))
def test_download_csv_dataset(region, year, level):
    if year == 2018 and level=="L2A":
        pytest.skip(msg="no csvfiles for 2018 L2A added, yet. skipping test")

    # run __init__
    ds = BreizhCrops(region=region, root=TESTS_DATA_ROOT, load_timeseries=True, level=level,
                     recompile_h5_from_csv=True, year=year)

    # run __getitem__
    sample = ds[0]

    # run __len__
    len(ds)
"""

@pytest.mark.parametrize("region,year,level",
                         product(["frh01", "frh02", "frh03", "frh04"],
                                 [2017, 2018],
                                 ["L1C", "L2A"]))
def test_download_h5dataset(region, year, level):
    if year == 2018:
        pytest.skip(msg="no h5databases for 2018 added, yet. skipping test")

    # run __init__
    ds = BreizhCrops(region=region, root=TESTS_DATA_ROOT, load_timeseries=True, level=level,
                     recompile_h5_from_csv=False, year=year)

    # run __getitem__
    sample = ds[0]
    X, y, fid = sample

    assert int(y) in list(ds.classes)

    assert X.numpy().min() >= 0, "min reflectances should be greater than 0"
    assert X.numpy().max() <= 1, "max reflectances should be smaller than 1"

    red = X[:, SELECTED_BANDS[level].index("B4")]
    nir = X[:, SELECTED_BANDS[level].index("B8")]

    ndvi = (nir - red) / (nir + red)

    assert ndvi.min() > 0
    assert ndvi.max() < 1

    # run __len__
    len(ds)

@pytest.mark.parametrize("region,year",
                         product(["frh01", "frh02", "frh03", "frh04"],
                                 [2017, 2018]))
def test_breizhcrops_geodataframe(region, year):
    BreizhCrops(region=region, root=TESTS_DATA_ROOT, year=year, level="L1C", load_timeseries=False).geodataframe()
