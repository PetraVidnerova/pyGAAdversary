#!/bin/bash

function wait {
    WAIT=true 

    while $WAIT 
    do 
	PS=`pgrep "python" | wc --lines`
	if test $PS -lt 12
	then
	    WAIT=false 
	else 
	    sleep 5m
	fi 
    done 
}

function run {
    wait
    nohup python main.py $1 $2 > $1_$2.log &
    sleep 1m 
}


for I in `seq 4 9`
do
    run SVM_rbf $I
done
run SVM_sigmoid 2 
run SVM_sigmoid 3 
run SVM_sigmoid 4 
run SVM_sigmoid 5 
run SVM_sigmoid 6 
run SVM_sigmoid 7 
run SVM_sigmoid 8 
run SVM_sigmoid 9
for I in `seq 0 9`
do
    run RBF $I 
done  
