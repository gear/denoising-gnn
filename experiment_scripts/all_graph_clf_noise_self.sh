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
for dname in IMDBBINARY IMDBMULTI MUTAG NCI1 \
             PROTEINS PTC REDDITBINARY REDDITMULTI5K
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
                       --filename "$dname"_result_fold_$fold \
                       --degree_as_tag \
        > ./logs/"$dname".log
        echo Done.
    done
done
#!/bin/bash # Run all datasets with the default parameters cd .. source activate graph # Best parameter in isolated default search batch_size=32 num_layers=5 lr=0.01 num_mlp_layers=2 hidden_dim=32 final_dropout=0 epochs=20 for self_rate in $(seq 0.05 0.05 0.95) do echo Probability for correct label: $self_rate for fold in 0 1 2 3 4 5 6 7 8 9 do echo Processing COLLAB at fold $fold ... python main.py --dataset COLLAB \ --epochs $epochs \ --batch_size $batch_size \ --num_layers $num_layers \ --lr $lr \ --num_mlp_layers $num_mlp_layers \ --hidden_dim $hidden_dim \ --fold_idx $fold \ --filename COLLAB_self_"$self_rate"_result_fold_$fold \ --degree_as_tag \ --corrupt_label \ --T "$self_rate" \ > ./logs/COLLAB_self_"$self_rate".log echo Done. done done