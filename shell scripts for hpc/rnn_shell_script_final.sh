#!/bin/bash
##Kør på gpu
#BSUB -q gpuv100
##Antal gpuer vi vil bruge. Kommenter ud hvis cpu.
#BSUB -gpu "num=1:mode=exclusive_process"

##Kør på cpu
##BSUB -q hpc

##Navn på job
#BSUB -J rnn_sensitivity
##Output fil
#BSUB -o output/rnn_sensitivity-%J.out
##Antal kerner
#BSUB -n 1
##Om kernerne må være på forskellige computere
#BSUB -R "span[hosts=1]"
##Ram pr kerne
#BSUB -R "rusage[mem=16GB]"
##Hvor lang tid må den køre hh:mm
#BSUB -W 23:50
##Email når jobbet starter
#BSUB -B
##og stopper
#BSUB -N

module purge
module load python3/3.6.2
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

for num_hidden in 2
do
	for hidden_size in 40
	do
		for pred_seq_len in 80
		do
			for loss in mse
			do
				for weight_decay in 0.001
				do
					for dropout in 0.1
					do
						for case in 3 2 1
						do
							for rnn_type in gru
							do
								for drop_cols in none
								do
						  			python3 rnn_hpc_final.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --rnn_type=$rnn_type --case=$case
								done
							done
						done
					done
				done
			done
		done
	done
done
