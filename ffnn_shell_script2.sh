#!/bin/bash
##Kør på gpu
#BSUB -q gpuk80
##Antal gpuer vi vil bruge. Kommenter ud hvis cpu.
#BSUB -gpu "num=1:mode=exclusive_process"

##Kør på cpu
##BSUB -q hpc

##Navn på job
#BSUB -J ffnn_sensitivity
##Output fil
#BSUB -o output/ffnn_sensitivity-%J.out
##Antal kerner
#BSUB -n 1
##Om kernerne må være på forskellige computere
#BSUB -R "span[hosts=1]"
##Ram pr kerne
#BSUB -R "rusage[mem=4GB]"
##Hvor lang tid må den køre hh:mm
#BSUB -W 19:00
##Email når jobbet starter
#BSUB -B
##og stopper
#BSUB -N

module purge
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

for num_hidden in 5
do
	for hidden_size in 10 20 40 80 160
	do
		for pred_seq_len in 25
		do
			for loss in mse
			do
				for weight_decay in 0.001
				do
					for dropout in 0.1
					do
						for case in 1 2 3
						do
							for drop_col in none
							do
								python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case --drop_cols=$drop_col
							done
						done
					done
				done
			done
		done
	done
done
