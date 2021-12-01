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
#BSUB -n 4
##Om kernerne må være på forskellige computere
#BSUB -R "span[hosts=1]"
##Ram pr kerne
#BSUB -R "rusage[mem=8GB]"
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

for num_hidden in 5
do
	for hidden_size in 70
	do
		for pred_seq_len in 25
		do
			for loss in mse l1
			do
				for weight_decay in 0
				do
					for dropout in 0
					do
						for case in 1 2 3
						do
							for rnn_type in gru
							do
							  python3 rnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --rnn_type=$rnn_type --case=$case
							done
						done
					done
				done
			done
		done
	done
done
