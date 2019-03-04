#!/bin/bash

# Run all datasets with the default parameters
cd ..
source activate graph
for dname in COLLAB IMDBBINARY IMDBMULTI MUTAG NCI1 PROTEINS PTC REDDITBINARY REDDITMULTI5K 
do
    echo Processing $dname ...
    python main.py --dataset $dname > ./logs/default_$dname.log
    echo Done.
done
