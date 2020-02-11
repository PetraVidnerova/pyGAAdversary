#!/bin/bash

function wait {
    WAIT=true 

    while $WAIT 
    do 
	PS=`ps x | grep "python main.py" | wc --lines`
	if test $PS -lt 28
	then
	    WAIT=false 
	else 
	    sleep 5m
	fi 
    done 
}

function run {
   wait
   nohup python main.py $1 $2 $3 $4 > $1_$2_$3_$4.log &
   sleep 5s
}



for ID in `seq 3 9`; do
    for TARGET in `seq 0 9`; do
	for IMAGE in `seq 0 9`; do
	    if test $TARGET -ne $IMAGE; then
		run CNN $TARGET $IMAGE $ID
	    fi
	done
    done 
done

