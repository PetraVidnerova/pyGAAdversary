#!/bin/bash

function wait {
    WAIT=true 

    while $WAIT 
    do 
	PS=`ps x | grep "python main.py" | wc --lines`
	if test $PS -lt 12
	then
	    WAIT=false 
	else 
	    sleep 5m
	fi 
    done 
}

function run {
#    wait
    nohup python main.py $1 $2 $3 $4 > $1_$2_$3_$4.log &
    sleep 5s
}


# run CNN 1 0 10.0 
# run CNN 1 0 5.0 
# run CNN 1 0 3.0 
# run CNN 1 0 1.0 

run CNN 1 0 0 
run CNN 1 2 0
run CNN 1 3 0
run CNN 1 4 0
run CNN 1 5 0 
run CNN 1 6 0
run CNN 1 7 0
run CNN 1 8 0
run CNN 1 9 0

run CNN 0 1 0 
run CNN 0 2 0
run CNN 0 3 0
run CNN 0 4 0
run CNN 0 5 0 
run CNN 0 6 0
run CNN 0 7 0
run CNN 0 8 0
run CNN 0 9 0

# run CNN 2 0 0 
# run CNN 2 1 0
# run CNN 2 3 0
# run CNN 2 4 0
# run CNN 2 5 0 
# run CNN 2 6 0
# run CNN 2 7 0
# run CNN 2 8 0
# run CNN 2 9 0


# run CNN 1 2 1.0 
# run CNN 1 2 2.0 
# run CNN 1 2 3.0 
# run CNN 1 2 4.0
# run CNN 1 2 5.0  
# run CNN 1 2 6.0 
# run CNN 1 2 7.0 
# run CNN 1 2 8.0 
# run CNN 1 2 9.0 
# run CNN 1 2 10.0 

# run CNN 1 3 1.0 
# run CNN 1 3 2.0 
# run CNN 1 3 3.0 
# run CNN 1 3 4.0
# run CNN 1 3 5.0  
# run CNN 1 3 6.0 
# run CNN 1 3 7.0 
# run CNN 1 3 8.0 
# run CNN 1 3 9.0 
# run CNN 1 3 10.0 


# run CNN 1 2 1.0 
# run CNN 1 3 1.0 
# run CNN 1 4 1.0 
# run CNN 1 5 1.0 
# run CNN 1 6 1.0 
# run CNN 1 7 1.0 
# run CNN 1 8 1.0 
# run CNN 1 9 1.0 

# run CNN 0 1 1.0
# run CNN 0 2 1.0
# run CNN 0 3 1.0
# run CNN 0 4 1.0
# run CNN 0 5 1.0
# run CNN 0 6 1.0
# run CNN 0 7 1.0
# run CNN 0 8 1.0
# run CNN 0 9 1.0

# run CNN 2 0 1.0
# run CNN 2 1 1.0
# run CNN 2 3 1.0
# run CNN 2 4 1.0
# run CNN 2 5 1.0
# run CNN 2 6 1.0
# run CNN 2 7 1.0
# run CNN 2 8 1.0
# run CNN 2 9 1.0

# run SVM_poly 2 0 1.0
# run SVM_poly 2 1 1.0
# run SVM_poly 2 3 1.0
# run SVM_poly 2 4 1.0
# run SVM_poly 2 5 1.0
# run SVM_poly 2 6 1.0
# run SVM_poly 2 7 1.0
# run SVM_poly 2 8 1.0
# run SVM_poly 2 9 1.0
