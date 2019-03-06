#!/bin/bash

# Run all datasets with the default parameters
cd ..
source activate graph
# Best parameter in isolated default search
batch_size=32
num_layers=5
lr=0.01
num_mlp_layers=2
hidden_dim=32
final_dropout=0
epochs=10
for est in min max
do
    for cor in backward forward compound
    do
        for c in estimate exact 
        do
            echo "====est: $est, cor: $cor, C: $c===="
            for rate in $(seq 0.05 0.1 0.95)
            do
                for fold in $(seq 0 1 9)
                do
                    python main.py --dataset MUTAG \
                                   --epochs $epochs \
                                   --batch_size $batch_size \
                                   --num_layers $num_layers \
                                   --lr $lr \
                                   --num_mlp_layers $num_mlp_layers \
                                   --hidden_dim $hidden_dim \
                                   --fold_idx $fold \
                                   --filename MUTAG_"$rate"_$fold \
                                   --corrupt_label \
                                   --N "$rate" \
                                   --denoise $c \
                                   --correction $cor \
                                   --est_mode $est \
                    > ./logs/MUTAG_denoise.log
                done
            done
        done
    done
done
