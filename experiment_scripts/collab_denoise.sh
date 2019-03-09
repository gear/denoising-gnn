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
epochs=20
for noise in $(seq 0.05 0.05 0.95)
do
for fold in $(seq 0 1 9)
do
    echo Processing MUTAG at fold $fold ...
    python main.py --dataset MUTAG \
                   --epochs $epochs \
                   --batch_size $batch_size \
                   --num_layers $num_layers \
                   --lr $lr \
                   --num_mlp_layers $num_mlp_layers \
                   --hidden_dim $hidden_dim \
                   --fold_idx $fold \
                   --filename MUTAG_"$noise"_exact_fold_$fold \
                   --degree_as_tag \
                   --corrupt_label \
                   --N "$noise" \
                   --denoise exact \
                   --correction backward \
    > ./logs/MUTAG_backward_exact.log
    echo Done.
done
done
