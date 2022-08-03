#!/usr/bin/env bash

#datasets='synthetic_1_1 synthetic_iid synthetic_0_0 synthetic_0.5_0.5 nist celeba'

datasets='celeba'
clmodel='cnn'

for dataset in $datasets
do
    if [ $dataset == 'synthetic_1_1' ]
    then
        L_auxs=( 35) #5 10 25 50
    elif [ $dataset == 'synthetic_0.5_0.5' ]
    then
        L_auxs=( 1 5 10 20)
    elif [ $dataset == 'synthetic_0_0' ]
    then
        L_auxs=( 1 3 7 10)
    else
        L_auxs=( 1 2 3 4)
    fi
    L_aux=1
    #for L_aux in "${L_auxs[@]}"
    for num_clients in 10 20
    do
        for epoch in 1 #20
        do
            for m in 1 10 # use any value other than 1 to run partial gradient setting
            do
                echo $L_aux
                python3  -u main.py --dataset=$dataset --optimizer='fedavg'  \
                --learning_rate=0.1 --num_rounds=800 --Ls0=$L_aux \
                --eval_every=1 --batch_size=10 \
                --num_epochs=$epoch \
                --model=$clmodel \
                --drop_percent=0 \
                --clients_per_round=$num_clients \
                --sim_metric='grad' --m_interval=$m \
                --clientsel_algo='submodular' | tee results/$dataset/uneq_submod_numclients$num_clients"epochs"$epoch"updateevery"$m"ICLR_stochgreedy_newpartition"
                #--clientsel_algo='lossbased' | tee results/$dataset/uneq_PoC_numclients$num_clients"epochs"$epoch"T1"
                #--clientsel_algo='lossbased' | tee results/$dataset/uneq11_simpleavg_PoC_numclients$num_clients"epochs"$epoch"T1"    
            done
        done
    done
done
echo All done