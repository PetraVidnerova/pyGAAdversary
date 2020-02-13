MODEL=$1

for ID in `seq 0 9`; do
    ./eval.sh $MODEL $ID > heatmap_$MODEL_$ID.log
done 
