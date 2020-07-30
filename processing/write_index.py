import sys
region = sys.argv[1]
sys.path.append("/home/ga63cuh/BreizhCrops_remote")
from breizhcrops import BreizhCrops

BreizhCrops(region=region, root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2018)
