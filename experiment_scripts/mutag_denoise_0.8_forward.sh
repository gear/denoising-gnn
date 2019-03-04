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
for fold in 0
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
                   --filename MUTAG_0.8_result_fold_$fold \
                   --degree_as_tag \
                   --corrupt_label \
                   --N "0.8" \
                   --denoise estimate \
                   --correction forward
    > ./logs/COLLAB_0.8_forward.log
    echo Done.
done
