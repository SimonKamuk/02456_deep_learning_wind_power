#!/bin/bash
##Kør på gpu
#BSUB -q gpuv100
##Antal gpuer vi vil bruge. Kommenter ud hvis cpu.
#BSUB -gpu "num=1:mode=exclusive_process"

##Kør på cpu
##BSUB -q hpc

##Navn på job
#BSUB -J ffnn_sensitivity
##Output fil
#BSUB -o output/ffnn_sensitivity-%J.out
##Antal kerner
#BSUB -n 4
##Om kernerne må være på forskellige computere
#BSUB -R "span[hosts=1]"
##Ram pr kerne
#BSUB -R "rusage[mem=4GB]"
##Hvor lang tid må den køre hh:mm
#BSUB -W 23:50
##Email når jobbet starter
#BSUB -B
##og stopper
#BSUB -N

module purge

for num_hidden in 1 5 10 15 20
do
	for hidden_size in 70
	do
		for pred_seq_len in 25
		do
			for loss in mse
			do
				for weight_decay in 0
				do
					for dropout in 0
					do
						for case in 1 2 3
						do
							python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case
						done
					done
				done
			done
		done
	done
done


for num_hidden in 5
do
	for hidden_size in 10 20 40 80 160
	do
		for pred_seq_len in 25
		do
			for loss in mse
			do
				for weight_decay in 0
				do
					for dropout in 0
					do
						for case in 1 2 3
						do
							python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case
						done
					done
				done
			done
		done
	done
done

for num_hidden in 5
do
	for hidden_size in 70
	do
		for pred_seq_len in 20 40 80 120 160
		do
			for loss in mse
			do
				for weight_decay in 0
				do
					for dropout in 0
					do
						for case in 1 2 3
						do
							python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case
						done
					done
				done
			done
		done
	done
done

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
							python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case
						done
					done
				done
			done
		done
	done
done

for num_hidden in 5
do
	for hidden_size in 70
	do
		for pred_seq_len in 25
		do
			for loss in mse
			do
				for weight_decay in 0 0.001 0.005 0.01 0.1
				do
					for dropout in 0
					do
						for case in 1 2 3
						do
							python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case
						done
					done
				done
			done
		done
	done
done


for num_hidden in 5
do
	for hidden_size in 70
	do
		for pred_seq_len in 25
		do
			for loss in mse
			do
				for weight_decay in 0
				do
					for dropout in 0 0.005 0.01 0.05 0.1
					do
						for case in 1 2 3
						do
							python3 ffnn_hpc.py --num_hidden=$num_hidden --hidden_size=$hidden_size --pred_seq_len=$pred_seq_len --loss=$loss --weight_decay=$weight_decay --dropout=$dropout --case=$case
						done
					done
				done
			done
		done
	done
done
