#!/bin/bash


function run {
    GEN=`tail -n 3 $1_$2_$3_$4.log | head -n 1 | sed 's/\t/;/g' | cut -d';' -f3`
    echo -n $GEN
}
MODEL=$1
ID=$2


for TARGET in `seq 0 9`; do
    for IMAGE in `seq 0 9`; do
	if test $TARGET -ne $IMAGE; then
	    run $MODEL $TARGET $IMAGE $ID
	else 
	    echo -n 0
	fi
	echo -n " "
    done
    echo
done 


