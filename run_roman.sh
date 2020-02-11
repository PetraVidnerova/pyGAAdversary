#!/bin/bash


function run {
   python main.py $1 $2 $3 $4 > $1_$2_$3_$4.log 
}

$ID=2

for TARGET in `seq 4 9`; do
    for IMAGE in `seq 0 9`; do
	if test $TARGET -ne $IMAGE; then
	    run CNN $TARGET $IMAGE $ID
	fi
    done
done 


