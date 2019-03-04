#!/bin/bash

# Run all datasets with the default parameters
cd ..
source activate graph
# Best parameter in isolated default search
batch_size=128
num_layers=2
lr=0.01
num_mlp_layers=1
hidden_dim=64
for fold in 0 1 2 3 4 5 6 7 8 9
do
    echo Processing MUTAG at number of MLP layer $num_mlp_layers ...
    python main.py --dataset MUTAG \
                   --batch_size $batch_size \
                   --num_layers $num_layers \
                   --lr $lr \
                   --num_mlp_layers $num_mlp_layers \
                   --hidden_dim $hidden_dim \
                   --fold_idx $fold \
                   --filename MUTAG_result_fold_$fold \
    > ./logs/MUTAG_bs_"$batch_size"_nl_"$num_layers"_lr_"$lr"_mlp_"$num_mlp_layers"_hd_"$hidden_dim"_fold_$fold.log
    echo Done.
done
