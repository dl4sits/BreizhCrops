#!/bin/bash

#~ python train_rf.py \
	#~ model RF \
	#~ -m evaluation \
	#~ -D /data/BreizhCrops/L2A_img/output/data/ \
	#~ -w 20 \
	#~ -l /data/BreizhCrops/L2A_img/output/RF_results/ \
	#~ --level L1C-interp \
	#~ --preload-ram

declare -a modes=("evaluation1" "evaluation2" "evaluation3" "evaluation4")
declare -a levels=("L1C-interp" "L2A-interp")

for mode in ${modes[@]}; do
	for level in ${levels[@]}; do
		python train_rf.py \
			model RF \
			-m evaluation \
			-D /data/BreizhCrops/L2A_img/output/data/ \
			-w 12 \
			-l /data/BreizhCrops/L2A_img/output/RF_results/ \
			--level $level \
			--mode $mode \
			--preload-ram
	done
done	
