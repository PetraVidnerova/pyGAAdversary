#!/bin/bash

N_PROC=10

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


# for MODEL in CNN MLP SVM_rbf SVM_linear  SVM_sigmoid SVM_poly SVM_poly4 RBF DT; d
#for MODEL in  CNN MLP "SVM_rbf" "SVM_linear"  "SVM_sigmoid" "SVM_poly" "SVM_poly4" RBF DT; do
for MODEL in RBF; do
    if $MODEL -eq "RBF"; then
	export N_PROC=10
    fi	
    for ID in "0_N0_9"; do
	for TARGET in `seq 0 9`; do
	    for IMAGE in `seq 0 9`; do
		if test $TARGET -ne $IMAGE; then
		    run $MODEL $TARGET $IMAGE $ID
		fi
	    done
	done 
    done
done

