#!/usr/bin/env bash

#datasets='synthetic_1_1 synthetic_iid synthetic_0_0 synthetic_0.5_0.5'
datasets='celeba'
clmodel='cnn'

for dataset in $datasets
do
    for num_clients in 10 20 #10 15 20
    do
        for epoch in 1 #5 10
        do
            for mu in 0 #1
            do
                echo $dataset $num_clients $epoch
                python3  -u main.py --dataset=$dataset --optimizer='fedprox'  \
                --learning_rate=0.1 --num_rounds=800 --clients_per_round=$num_clients \
                --eval_every=1 --batch_size=10 \
                --num_epochs=$epoch \
                --model=$clmodel \
                --drop_percent=0 \
                --mu=$mu | tee results/$dataset/fedprox_numclients$num_clients"mu"$mu"epochs"$epoch"ICLR"
            done
        done
    done
done

echo All done  