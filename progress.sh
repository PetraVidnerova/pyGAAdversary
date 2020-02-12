MODEL=$1
for ID in `seq 0 9`; do 
    echo -n "$ID: "
    ls -1 adversary_sample_${MODEL}_?_?_$ID.npy  2>/dev/null| wc --lines
done

