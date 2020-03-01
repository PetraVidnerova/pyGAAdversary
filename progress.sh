#for ID in "0_N0_5";  do 
#1;c1;5202;0c    echo "$ID: "
for MODEL in CNN MLP SVM_rbf SVM_linear  SVM_sigmoid SVM_poly SVM_poly4 RBF DT; do
    echo -n "$MODEL: "
    for ID in "0_N0_1" "0_N0_2" "0_N0_3" "0_N0_4" "0_N0_5" "0_N0_6" "0_N0_7" "0_N0_8" "0_N0_9"; do
	NUM=`ls -1 adversary_sample_${MODEL}_?_?_$ID.npy  2>/dev/null| wc --lines`
	echo -n "$NUM  "
    done
    echo
done
#done

