
for ID in "0_N0_5";  do 
    echo "$ID: "
    for MODEL in CNN MLP SVM_rbf SVM_linear  SVM_sigmoid SVM_poly SVM_poly4 RBF DT; do
	echo -n "$MODEL: "
	ls -1 adversary_sample_${MODEL}_?_?_$ID.npy  2>/dev/null| wc --lines
    done
done

