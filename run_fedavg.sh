#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=5 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=1 \
            --model='mclr' \
            --drop_percent=$2 \
            --clientsel_algo=='submodular' \


