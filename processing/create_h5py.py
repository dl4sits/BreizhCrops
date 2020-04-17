from breizhcrops import BreizhCrops

datapath = "/home/marc/remote/BreizhCrops/data"
transform = None
target_transform = None

BreizhCrops(region="frh04", root=datapath, load_timeseries=False)
BreizhCrops(region="frh01", root=datapath, load_timeseries=False)
BreizhCrops(region="frh02", root=datapath, load_timeseries=False)
BreizhCrops(region="frh03", root=datapath, load_timeseries=False)