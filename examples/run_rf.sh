#!/bin/bash

python train_rf.py \
	model RF \
	-m evaluation \
	-D /data/BreizhCrops/L2A_img/output/data/ \
	-w 12 \
	--level L2A-interp \
	--preload-ram
	
