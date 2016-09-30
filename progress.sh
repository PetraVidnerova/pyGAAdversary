for MODEL in "MLP" "CNN" "DT" "RBF" "SVM_rbf" "SVM_linear" "SVM_sigmoid" "SVM_poly" "SVM_poly4"
do
    echo -n "$MODEL "
    for I in `seq 0 9`
    do
	if [ -e "adversary_inputs_against_${MODEL}_$I.npy" ]
	then
	    echo -n "$I "
	else 
	    if test `ps x | grep "$MODEL $I" | wc --lines` -gt 1
	    then 
		echo -n "r"
	    else
		echo -n "-"
	    fi
	fi
    done 
    echo 
done

