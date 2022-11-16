# !bin/bash

device=2
alg="SOP" # sop/rnn

declare -a testyear=( '2019' )
declare -a seeds=( 1234 )
declare -a lambdas=( 0.1 0.5 1.0 2.0 3.0 5.0 10.0 )
declare -a epsilons=( 0.1 0.4 0.5 1.0 1.5 )
declare -a deltas=( 0.05 0.1 0.2 )

for seed in "${seeds[@]}"
do
for y in "${testyear[@]}"
do
for l in "${lambdas[@]}"
do
for ep in "${epsilons[@]}"
do 
for d in "${deltas[@]}"
do
            python -u train_ili_seldonian.py \
            -d $device \
            -y $y \
            -w 1 \
            -a "trans" \
            -e 2000 \
            --model_name $alg \
            --eps $ep \
            --lamda $l \
            --delta $d \
            -s $seed \
            --constraint_pairs "[['7', '8'], ['6', '9'], ['5', '8'], ['4', '8'], ['1', '8'], ['6', '10'], ['3', '8'], ['8', '9'], ['2', '8'], ['8', '10']]" \
            --savedir "sop"
done
done 
done
done
done