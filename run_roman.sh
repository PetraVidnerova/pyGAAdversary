#!/bin/bash


function run {
#    wait
    python main.py $1 $2 $3 $4 > $1_$2_$3_$4.log 
#    sleep 5s
}

run CNN 1 0 1 
run CNN 1 2 1
run CNN 1 3 1
run CNN 1 4 1
run CNN 1 5 1 
run CNN 1 6 1
run CNN 1 7 1
run CNN 1 8 1
run CNN 1 9 1

run CNN 0 1 1 
run CNN 0 2 1
run CNN 0 3 1
run CNN 0 4 1
run CNN 0 5 1 
run CNN 0 6 1
run CNN 0 7 1
run CNN 0 8 1
run CNN 0 9 1

run CNN 2 0 1 
run CNN 2 1 1
run CNN 2 3 1
run CNN 2 4 1
run CNN 2 5 1 
run CNN 2 6 1
run CNN 2 7 1
run CNN 2 8 1
run CNN 2 9 1


