import sys
sys.path.append("..")

from breizhcrops import BreizhCrops
datapath = "/home/marc/remote/BreizhCrops/data"
transform = None
target_transform = None

BreizhCrops(region=sys.argv[1], root=datapath,level=sys.argv[2], load_timeseries=True, recompile_h5_from_csv=True)
#BreizhCrops(region="frh01", root=datapath, load_timeseries=True, recompile_h5_from_csv=True)
#BreizhCrops(region="frh02", root=datapath, load_timeseries=True, recompile_h5_from_csv=True)
#BreizhCrops(region="frh03", root=datapath, load_timeseries=True, recompile_h5_from_csv=True)