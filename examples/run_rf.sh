#!/bin/bash

#~ python train_rf.py \
	#~ model RF \
	#~ -m evaluation \
	#~ -D /data/BreizhCrops/L2A_img/output/data/ \
	#~ -w 20 \
	#~ -l /data/BreizhCrops/L2A_img/output/RF_results/ \
	#~ --level L1C-interp \
	#~ --preload-ram

python train_rf.py \
	model RF \
	-m evaluation \
	-D /data/BreizhCrops/L2A_img/output/data/ \
	-w 12 \
	-l /data/BreizhCrops/L2A_img/output/RF_results/ \
	--level L2A-interp \
	--preload-ram
	
