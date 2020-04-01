#!/bin/bash

L=40
NOW=$(date +"%m-%d-%y-%H-%M")

f1="L_$L"
f3="_$NOW.txt"
FILE=$f1$f3
Dir="log_files/"

unset DISPLAY
# matlab -nodesktop -nodisplay < pp_critical_txt_sparse.m &> $FILE  &
python3 cond_dis.py 2&>1 | tee $Dir$FILE
