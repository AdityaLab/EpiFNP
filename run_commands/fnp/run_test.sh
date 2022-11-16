# !bin/bash

device="cpu"
alg="FNP" # sop/rnn

declare -a testyear=( '2019' )
declare -a seeds=( 1234 )
declare -a lambdas=( 1.0 )
declare -a epsilons=( 0.5 )
declare -a deltas=( 0.05 )


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
            -e 3000 \
            --model_name $alg \
            --eps $ep \
            --lamda $l \
            --delta $d \
            -s $seed \
            --savedir "init_3000_numeval40"\
            --chkp_pattern "model_chkp_report/modelFNP_numConstraintAll_combineFalse_testyear2017_weekahead1_eps0.5_delta0.05_lambda1.0_epoch3000_seed1234*.pth"
done
done 
done
done
done