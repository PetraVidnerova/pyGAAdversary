#!/bin/bash

N_PROC=28

function wait {
    WAIT=true 

    while $WAIT 
    do 
	PS=`ps x | grep "python main.py" | wc --lines`
	if test $PS -lt $N_PROC
	then
	    WAIT=false 
	else 
	    sleep 1m
	fi 
    done 
}

function run {
   wait
   nohup python main.py $1 $2 $3 $4 > $1_$2_$3_$4.log &
   sleep 1s
}



for ID in `seq 0 9`; do
    for TARGET in `seq 0 9`; do
	for IMAGE in `seq 0 9`; do
	    if test $TARGET -ne $IMAGE; then
		run MLP $TARGET $IMAGE $ID
	    fi
	done
    done 
done

