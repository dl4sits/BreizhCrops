import sys
import os
import time
sys.path.append("..")

import argparse
import h5py
from tqdm import tqdm

import breizhcrops

import numpy as np
import pandas as pd


import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def train(args):
	save_model = False

	logdir = os.path.join(args.logdir, "RF")
	os.makedirs(logdir, exist_ok=True)
	print(f"Logging results to {logdir}")

	X_train, y_train, X_test, y_test = get_dataloader(args.datapath, args.mode, args.preload_ram, args.level)

	rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
					max_depth=25, min_samples_split=2, oob_score=True, n_jobs=args.workers, verbose=1)
	
	#-- train a rf classifier
	start_traintime = time.time()				
	rf.fit(X_train, y_train)
	traintime = round(time.time()-start_traintime, 2)
	print('Training time (s): ', traintime)
			
	#-- save the model
	if save_model:
		joblib.dump(rf, model_file)
		print("Writing the model over")
			
	#-- prediction
	start_testtime =  time.time()
	y_pred = rf.predict(X_test)
	testtime = round(time.time()-start_testtime, 2)
	print('Test time (s): ', testtime)
			
	#p_test =  rf.predict_proba(X_test)
	scores = metrics(y_test, y_pred)
	scores["traintest"] = traintime
	scores["testtime"] = testtime
	scores["oob"] = rf.oob_score_
	scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
	print(scores_msg)
	
	log = list()
	log.append(scores)
	log_df = pd.DataFrame(log)
	log_df.to_csv(os.path.join(logdir, "resultlog.csv"))
	

def get_dataloader(datapath, mode, preload_ram=False, level="L1C"):
	print(f"Setting up datasets in {os.path.abspath(datapath)}, level {level}")
	datapath = os.path.abspath(datapath)

	padded_value = None
	sequencelength = 45

	bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
				'B8A', 'B11', 'B12', 'doa']
	selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
	selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

	def transform(x):
		#-- x = x[x[:, 0] != padded_value, :]  # remove padded values
		# choose selected bands
		#--- nor normalization required for RF
		#x = x[:, selected_band_idxs] * 1e-4  # scale reflectances to 0-1
		x = x 
		return x

	def target_transform(y):
		return y

	def get_data(areas):
		totalsize = 0
		for a in areas:
			totalsize = totalsize + a.index.shape[0]
		print("Total size: ", totalsize)

		X = []
		y = []
		for a in areas:
			for row in a:
				with h5py.File(a.h5path, "r") as dataset:
					X.append(np.array(row[0]))
				y.append(row[1])
		X = np.array(X)
		X = np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2]))
		y = np.array(y)
		print("X.shape: ", X.shape)
		print("y.shape: ", y.shape)
		return X, y
	
	if 0:
		#-- debug test
		frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath,  transform=transform, load_timeseries=True,
										target_transform=target_transform, padding_value=padded_value,
										preload_ram=preload_ram, level=level, recompile_h5_from_csv=True)
		testareas = [frh02]
		X_test, y_test = get_data(testareas)
		u, counts = np.unique(y_test, return_counts=True)
		
		print("u: ", u)
		print("counts: ", counts)
		print(1/0)
	
	
	frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath, transform=transform, load_timeseries=True,
									target_transform=target_transform, padding_value=padded_value,
									preload_ram=preload_ram, level=level, recompile_h5_from_csv=True )
	frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath,  transform=transform, load_timeseries=True,
									target_transform=target_transform, padding_value=padded_value,
									preload_ram=preload_ram, level=level, recompile_h5_from_csv=True)
	frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath, transform=transform, load_timeseries=True,
									target_transform=target_transform, padding_value=padded_value,
									preload_ram=preload_ram, level=level, recompile_h5_from_csv=True)
	if mode == "evaluation":
		frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath, transform=transform, load_timeseries=True,
									target_transform=target_transform, padding_value=padded_value, 
									preload_ram=preload_ram, level=level, recompile_h5_from_csv=True)
		trainareas = [frh01, frh02, frh03]
		X_train, y_train = get_data(trainareas)
		testareas = [frh04]
		X_test, y_test = get_data(testareas)
	elif mode == "validation":
		trainareas = [frh01, frh02]   
		X_train, y_train = get_data(trainareas)
		testareas = [frh03]
		X_test, y_test = get_data(testareas)
	else:
		raise ValueError("only --mode 'validation' or 'evaluation' allowed")

	
	meta = dict()
	if 0:
		meta = dict(
			ndims=len(selected_bands),
			num_classes=len(frh01.classes),
			sequencelength=sequencelength
		)

	return X_train, y_train, X_test, y_test

def metrics(y_true, y_pred):
	accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
	kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
	f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
	f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
	f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
	recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
	recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
	recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
	precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
	precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
	precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

	return dict(
		accuracy=accuracy,
		kappa=kappa,
		f1_micro=f1_micro,
		f1_macro=f1_macro,
		f1_weighted=f1_weighted,
		recall_micro=recall_micro,
		recall_macro=recall_macro,
		recall_weighted=recall_weighted,
		precision_micro=precision_micro,
		precision_macro=precision_macro,
		precision_weighted=precision_weighted,
	)

def parse_args():
	parser = argparse.ArgumentParser(description='Train and evaluate a traiditonal classificaiton algorithms on time series on the'
												'BreizhCrops dataset. This script trains a model on training dataset'
												'partition, evaluates performance on a validation or evaluation partition'
												'and stores progress and model paths in --logdir')
	parser.add_argument(
		'model', type=str, default="RF", help='select traditional algorithm. Available models are '
												'"RF"')
	parser.add_argument(
		'-m', '--mode', type=str, default="validation", help='training mode. Either "validation" '
												'(train on FRH01+FRH02 test on FRH03) or '
												'"evaluation" (train on FRH01+FRH02+FRH03 test on FRH04)')
	parser.add_argument(
		'-D', '--datapath', type=str, default="../data", help='directory to download and store the dataset')
	parser.add_argument(
		'-w', '--workers', type=int, default=0, help='number of CPU workers (parallel algorithm)')
	parser.add_argument(
		'-H', '--hyperparameter', type=str, default=None, help='model specific hyperparameter as single string, '
																'separated by comma of format param1=value1,param2=value2')
	parser.add_argument(
		'--level', type=str, default="L1C", help='Sentinel 2 processing level (L1C, L2A, L2A-interp)')
	parser.add_argument(
		'--preload-ram', action='store_true', help='load dataset into RAM upon initialization')
	parser.add_argument(
		'-l', '--logdir', type=str, default="/tmp", help='logdir to store progress and models (defaults to /tmp)')
	args, _ = parser.parse_known_args()

	hyperparameter_dict = dict()
	if args.hyperparameter is not None:
		for hyperparameter_string in args.hyperparameter.split(","):
			param, value = hyperparameter_string.split("=")
			hyperparameter_dict[param] = float(value) if '.' in value else int(value)
	args.hyperparameter = hyperparameter_dict

	return args


if __name__ == "__main__":
    args = parse_args()
	
    train(args)
