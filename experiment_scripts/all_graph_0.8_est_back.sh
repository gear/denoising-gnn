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

echo "====BIO-DATASETS===="
for dname in REDDITMULTI5K
do
    echo Processing $dname...
    for fold in 0 1 2 3 4 5 6 7 8 9
    do
        echo Processing $dname at fold $fold ...
        python main.py --dataset $dname \
                       --epochs $epochs \
                       --batch_size $batch_size \
                       --num_layers $num_layers \
                       --lr $lr \
                       --num_mlp_layers $num_mlp_layers \
                       --hidden_dim $hidden_dim \
                       --fold_idx $fold \
                       --filename "$dname"_0.8_$fold \
                       --corrupt_label \
                       --N 0.8 \
                       --denoise estimate \
                       --est_mode min \
        > ./logs/"$dname".log
        echo Done.
    done
done

echo "====SOCIAL-DATASETS===="
for dname in COLLAB
do
    echo Processing $dname...
    for fold in 0 1 2 3 4 5 6 7 8 9
    do
        echo Processing $dname at fold $fold ...
        python main.py --dataset $dname \
                       --epochs $epochs \
                       --batch_size $batch_size \
                       --num_layers $num_layers \
                       --lr $lr \
                       --num_mlp_layers $num_mlp_layers \
                       --hidden_dim $hidden_dim \
                       --fold_idx $fold \
                       --filename "$dname"_0.8_$fold \
                       --degree_as_tag \
                       --corrupt_label \
                       --N 0.8 \
                       --denoise estimate \
                       --est_mode min \
        > ./logs/"$dname".log
        echo Done.
    done
done
