#!/usr/bin/env bash

#SBATCH --job-name=graphormer                                   # sets the job name if not set from environment
#SBATCH --array=1-4                                           # Submit ##-##%## array jobs inclusive, throttling to %## at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                        # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                         # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=4:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                                     # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 32gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=TIME_LIMIT,FAIL,ARRAY_TASKS                 # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

##### For now exclude >= 17 as well as 8,11 because those are the A4000 nodes and require higher cuda version
#SBATCH --exclude=cml08,cml11,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24

function runexp {

gpu=${1}
dataset=${2}
num_layers=${3}
hidden_channels=${4}
minibatch=${5}
method=${6}

echo "dataset: " $dataset

if [ "$dataset" = "fb100" ]; then
    sub_dataset="Penn94"
    echo "sub_dataset: " $sub_dataset
fi

mkdir -p stdout-logs

expname=${method}-${minibatch}-${dataset}${sub_dataset}-${num_layers}-${hidden_channels}

echo stdout-logs/${expname}.log
echo "output log for ${expname}" > stdout-logs/${expname}.log

if [ "$minibatch" = "fullbatch" ]; then
    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
        CUDA_VISIBLE_DEVICES=$gpu \
        python main.py --dataset $dataset \
                    --sub_dataset ${sub_dataset:-''} \
                    --method linkx \
                    --num_layers $num_layers \
                    --hidden_channels $hidden_channels \
                    --display_step 25 \
                    --runs 5 \
                    --directed \
                    > stdout-logs/${expname}.log 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu \
        python main.py --dataset $dataset \
                    --sub_dataset ${sub_dataset:-''} \
                    --method linkx \
                    --num_layers $num_layers \
                    --hidden_channels $hidden_channels \
                    --display_step 25 \
                    --runs 5 \
                    > stdout-logs/${expname}.log 2>&1
    fi
else # minibatch
    if [ "$method" = "linkx" ]; then
        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset \
                                    --sub_dataset ${sub_dataset:-''} \
                                    --method linkx \
                                    --num_layers $num_layers \
                                    --hidden_channels $hidden_channels \
                                    --display_step 25 \
                                    --runs 5 \
                                    --train_batch row \
                                    --num_parts 10 \
                                    --directed \
                                    > stdout-logs/${expname}.log 2>&1
        else
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset \
                                    --sub_dataset ${sub_dataset:-''} \
                                    --method linkx \
                                    --num_layers $num_layers \
                                    --hidden_channels $hidden_channels \
                                    --display_step 25 \
                                    --runs 5 \
                                    --train_batch row \
                                    --num_parts 10 \
                                    > stdout-logs/${expname}.log 2>&1
        fi
    else # method=mixhop-saintrw-#k
        if [ "$method" = "mixhop-saintrw-5k" ]; then
            batch_size=5000
        elif [ "$method" = "mixhop-saintrw-10k" ]; then
            batch_size=10000
        else
            echo "Error! Unexpected method=$method " > stdout-logs/${expname}.log
            exit 1
        fi

        if [ "$dataset" = "arxiv-year" ] || [ "$dataset" = "genius" ] || [ "$dataset" = "fb100" ]; then
            # echo "Successful dry launch of $method w/ $minibatch batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method mixhop --hops 2 --num_layers $num_layers --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                    --no_mini_batch_test --saint_num_steps 5 \
                    > stdout-logs/${expname}.log 2>&1
        else
            # echo "Successful dry launch of $method w/ $minibatch batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method mixhop --hops 2 --num_layers $num_layers --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                    --saint_num_steps 5 --test_num_parts 10 \
                    > stdout-logs/${expname}.log 2>&1
        fi
    fi
fi

}

set | grep SLURM_JOB_NODELIST | while read line; do echo "$line"; done

eval "$(conda shell.bash hook)"
source /cmlscratch/jkirchen/graphormer-root/P4-Graph-Transformer/set_env.sh

# debugging lists

# dataset_lst=( "wiki" )
# num_layers_lst=( 1 2 )
# hidden_channels_lst=( 16 32 )
# minibatch_lst=( "fullbatch" )

# Full sweep array

# dataset_lst=( "arxiv-year" "fb100" "twitch-gamer" "genius" "pokec" "snap-patents" "wiki" )
# num_layers_lst=( 1 2 3 )
# hidden_channels_lst=( 16 32 128 256 )
# minibatch_lst=( "fullbatch" "minibatch" )

# Subset of sweep array runnable on 2080s (except maybe one or two settings)

# dataset_lst=( "arxiv-year" "fb100" "twitch-gamer" "genius" "pokec" "ogbn-arxiv" )
# num_layers_lst=( 1 2 3 4 )
# hidden_channels_lst=( 64 128 256 512 )
# minibatch_lst=( "minibatch" )
# method_lst=( "linkx" )

# debugging ogbn datasets

# dataset_lst=( "ogbn-arxiv" )
# num_layers_lst=( 1 2 3 4 )
# hidden_channels_lst=( 64 128 256 512 )
# minibatch_lst=( "minibatch" )

# mixhop init baseline sweep
# dataset_lst=( "arxiv-year" "fb100" "twitch-gamer" "genius" "pokec" )
# num_layers_lst=( 2 4 )
# hidden_channels_lst=( 128 )
# minibatch_lst=( "minibatch" )
# method_lst=( "mixhop-saintrw-5k" "mixhop-saintrw-10k" )

# doing the mixhup ogbn-arxiv run
dataset_lst=( "ogbn-arxiv" )
num_layers_lst=( 2 4 )
hidden_channels_lst=( 128 )
minibatch_lst=( "minibatch" )
method_lst=( "mixhop-saintrw-5k" "mixhop-saintrw-10k" )

i_d=$(( (${SLURM_ARRAY_TASK_ID} - 1)  % ${#dataset_lst[@]} ))
i_nl=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} % ${#num_layers_lst[@]} ))
i_hc=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} % ${#hidden_channels_lst[@]} ))
i_mb=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} / ${#hidden_channels_lst[@]} % ${#minibatch_lst[@]}))
i_me=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} / ${#hidden_channels_lst[@]} / ${#minibatch_lst[@]} % ${#method_lst[@]}))


# runexp    gpu   dataset                  num_layers                   hidden_channels                    minibatch                    method                 
runexp      0     ${dataset_lst[${i_d}]}   ${num_layers_lst[${i_nl}]}   ${hidden_channels_lst[${i_hc}]}    ${minibatch_lst[${i_mb}]}    ${method_lst[${i_me}]}


