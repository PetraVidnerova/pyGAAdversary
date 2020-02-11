#!/bin/bash


function run {
   python main.py $1 $2 $3 $4 > $1_$2_$3_$4.log 
}


for ID in `seq 0 9`; do
    for TARGET in `seq 0 9`; do
	for IMAGE in `seq 0 9`; do
	    if test $TARGET -ne $IMAGE; then
		run DT $TARGET $IMAGE $ID
	    fi
	done
    done 
done

