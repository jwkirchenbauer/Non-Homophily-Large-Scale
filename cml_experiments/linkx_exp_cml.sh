#!/usr/bin/env bash

#SBATCH --job-name=graphormer                                   # sets the job name if not set from environment
#SBATCH --array=1-4                                          # Submit ##-##%## array jobs inclusive, throttling to %## at a time
#SBATCH --output slurm-logs/%x_%A_%a.log                        # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error slurm-logs/%x_%A_%a.log                         # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=20:00:00                                          # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                                     # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem 48gb                                              # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=TIME_LIMIT,FAIL,ARRAY_TASKS                 # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE,

##### For now exclude >= 17 as well as 8,11 because those are the A4000 nodes and require higher cuda version
### #SBATCH --exclude=cml08,cml11,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24

function runexp {

gpu=${1}
dataset=${2}
num_layers=${3}
hidden_channels=${4}
minibatch=${5}
method=${6}
train_prop=${7:-"0.5"}
valid_prop=${8:-"0.25"}
gat_heads=${9:-"None"}
lr=${10:-"None"}

echo "dataset: " $dataset

if [ "$dataset" = "fb100" ]; then
    sub_dataset="Penn94"
    echo "sub_dataset: " $sub_dataset
fi

mkdir -p stdout-logs

expname=${method}-${minibatch}-${dataset}${sub_dataset}-${num_layers}-${hidden_channels}-${train_prop}-${valid_prop}-${gat_heads}-${lr}

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
                    --train_prop $train_prop \
                    --valid_prop $valid_prop \
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
                    --train_prop $train_prop \
                    --valid_prop $valid_prop \
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
                                    --train_prop $train_prop \
                                    --valid_prop $valid_prop \
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
                                    --train_prop $train_prop \
                                    --valid_prop $valid_prop \
                                    > stdout-logs/${expname}.log 2>&1
        fi
    elif [ "$method" = "mixhop-saintrw-5k" ] || [ "$method" = "mixhop-saintrw-10k" ]; then # method=mixhop-saintrw-#k
        if [ "$method" = "mixhop-saintrw-5k" ]; then
            batch_size=5000
        elif [ "$method" = "mixhop-saintrw-10k" ]; then
            batch_size=10000
        else
            echo "Error! Unexpected method=$method " > stdout-logs/${expname}.log
            exit 1
        fi

        if [ "$dataset" = "ogbn-arxiv" ] || [ "$dataset" = "arxiv-year" ] || [ "$dataset" = "genius" ] || [ "$dataset" = "fb100" ]; then
            # echo "Successful dry launch of $method w/ $minibatch batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method mixhop --hops 2 --num_layers $num_layers --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                    --no_mini_batch_test --saint_num_steps 5 \
                    --train_prop $train_prop \
                    --valid_prop $valid_prop \
                    > stdout-logs/${expname}.log 2>&1
        else
            # echo "Successful dry launch of $method w/ $minibatch batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                    --method mixhop --hops 2 --num_layers $num_layers --hidden_channels $hidden_channels \
                    --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                    --saint_num_steps 5 --test_num_parts 10 \
                    --train_prop $train_prop \
                    --valid_prop $valid_prop \
                    > stdout-logs/${expname}.log 2>&1
        fi
    elif [ "$method" = "mixhop-cluster-200-1" ] || [ "$method" = "mixhop-cluster-200-5" ]; then # method=mixhop-cluster
        if [ "$method" = "mixhop-cluster-200-1" ]; then
            num_parts=200
            cluster_batch_size=1
        elif [ "$method" = "mixhop-cluster-200-5" ]; then
            num_parts=200
            cluster_batch_size=5
        else
            echo "Error! Unexpected method=$method " > stdout-logs/${expname}.log
            exit 1
        fi
        # echo "Successful dry launch of $method w/ num_parts=$num_parts cluster_batch_size=$cluster_batch_size layers=$num_layers channels=$hidden_channels" > stdout-logs/${expname}.log
        CUDA_VISIBLE_DEVICES=$gpu \
        python main_scalable.py --dataset $dataset \
                        --method mixhop --num_layers $num_layers \
                        --hidden_channels $hidden_channels \
                        --display_step 25 --runs 3 --hops 2 \
                        --train_batch cluster --num_parts $num_parts \
                        --cluster_batch_size $cluster_batch_size \
                        > stdout-logs/${expname}.log 2>&1

    elif [ "$method" = "gcnjk-saintrw-5k" ] || [ "$method" = "gcnjk-saintrw-10k" ]; then
        if [ "$method" = "gcnjk-saintrw-5k" ]; then
            batch_size=5000
        elif [ "$method" = "gcnjk-saintrw-10k" ]; then
            batch_size=10000
        else
            echo "Error! Unexpected method=$method " > stdout-logs/${expname}.log
            exit 1
        fi
        if [ "$dataset" = "ogbn-arxiv" ] || [ "$dataset" = "arxiv-year" ] || [ "$dataset" = "genius" ] || [ "$dataset" = "fb100" ]; then
            # echo "Successful dry launch of $method on $dataset batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                --method gcnjk --jk_type cat --num_layers $num_layers --hidden_channels $hidden_channels \
                --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                --no_mini_batch_test --saint_num_steps 5 \
                --train_prop $train_prop \
                --valid_prop $valid_prop \
                > stdout-logs/${expname}.log 2>&1
        else
            # echo "Successful dry launch of $method on $dataset batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                --method gcnjk --jk_type cat --num_layers $num_layers --hidden_channels $hidden_channels \
                --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size $batch_size \
                --saint_num_steps 5 --test_num_parts 10 \
                --train_prop $train_prop \
                --valid_prop $valid_prop \
                > stdout-logs/${expname}.log 2>&1
        fi
    elif [ "$method" = "gat-saintrw-2p5k" ] || [ "$method" = "gat-saintrw-5k" ] || [ "$method" = "gat-saintrw-10k" ]; then
        if [ "$method" = "gat-saintrw-2p5k" ]; then
            batch_size=2500
        elif [ "$method" = "gat-saintrw-5k" ]; then
            batch_size=5000
        elif [ "$method" = "gat-saintrw-10k" ]; then
            batch_size=10000
        else
            echo "Error! Unexpected method=$method " > stdout-logs/${expname}.log
            exit 1
        fi
        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
            # echo "Successful dry launch of $method with $dataset $gat_heads $lr batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                --method gat --num_layers $num_layers --hidden_channels $hidden_channels \
                --lr $lr --gat_heads $gat_heads --directed --display_step 25 --runs 5 \
                --train_batch graphsaint-rw --batch_size $batch_size \
                --saint_num_steps 5 --test_num_parts 10 \
                --train_prop $train_prop \
                --valid_prop $valid_prop \
                > stdout-logs/${expname}.log 2>&1
        elif [ "$dataset" = "ogbn-arxiv" ]; then
            # echo "Successful dry launch of $method with $dataset $gat_heads $lr batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                --method gat --num_layers $num_layers --hidden_channels $hidden_channels \
                --lr $lr --gat_heads $gat_heads --display_step 25 --runs 5 \
                --train_batch graphsaint-rw --batch_size $batch_size \
                --saint_num_steps 5 \
                --no_mini_batch_test \
                --train_prop $train_prop \
                --valid_prop $valid_prop \
                > stdout-logs/${expname}.log 2>&1
        else
            # echo "Successful dry launch of $method with $dataset $gat_heads $lr batch_size=$batch_size" > stdout-logs/${expname}.log
            CUDA_VISIBLE_DEVICES=$gpu \
            python main_scalable.py --dataset $dataset --sub_dataset ${sub_dataset:-''} \
                --method gat --num_layers $num_layers --hidden_channels $hidden_channels \
                --lr $lr --gat_heads $gat_heads --display_step 25 --runs 5 \
                --train_batch graphsaint-rw --batch_size $batch_size \
                --saint_num_steps 5 --test_num_parts 10 \
                --train_prop $train_prop \
                --valid_prop $valid_prop \
                > stdout-logs/${expname}.log 2>&1
        fi
    else
        echo "Error! Unexpected method=$method " > stdout-logs/${expname}.log
    fi
fi

}

set | grep SLURM_JOB_NODELIST | while read line; do echo "$line"; done

eval "$(conda shell.bash hook)"
# might want to switch auto based on gpu acquired
source /cmlscratch/jkirchen/graphormer-root/P4-Graph-Transformer/set_env.sh 113

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
# dataset_lst=( "ogbn-arxiv" )
# num_layers_lst=( 2 4 )
# hidden_channels_lst=( 128 )
# minibatch_lst=( "minibatch" )
# method_lst=( "mixhop-saintrw-5k" "mixhop-saintrw-10k" )

# Running snap-patents first for linkx then mixhop
# dataset_lst=( "snap-patents" )
# num_layers_lst=( 1 2 3 4 )
# hidden_channels_lst=( 16 32 128 256 )
# minibatch_lst=( "minibatch" )
# method_lst=( "linkx" )

# dataset_lst=( "snap-patents" )
# num_layers_lst=( 1 2 3 )
# hidden_channels_lst=( 16 32 128 )
# minibatch_lst=( "minibatch" )
# method_lst=( "mixhop-saintrw-5k" "mixhop-saintrw-10k" )

# Running ogbn-products with dataset.py code change
# dataset_lst=( "ogbn-products" )
# num_layers_lst=( 1 2 3 4 )
# hidden_channels_lst=( 16 32 128 256 )
# minibatch_lst=( "minibatch" )
# method_lst=( "linkx" )

# dataset_lst=( "ogbn-products" )
# num_layers_lst=( 1 2 3 )
# hidden_channels_lst=( 16 32 128 )
# minibatch_lst=( "minibatch" )
# method_lst=( "mixhop-saintrw-5k" "mixhop-saintrw-10k" )

# trying to get all gcnjk numbers
# dataset_lst=( "ogbn-arxiv" "ogbn-products" "arxiv-year" "snap-patents" )
# num_layers_lst=( 2 3 4) # 1 causes a matmul error
# hidden_channels_lst=( 16 32 128 256)
# minibatch_lst=( "minibatch" )
# method_lst=( "gcnjk-saintrw-5k" "gcnjk-saintrw-10k" )

# trying for main gat numbers
# dataset_lst=( "ogbn-arxiv" "ogbn-products" "arxiv-year" "snap-patents" )
# num_layers_lst=( 2 )
# hidden_channels_lst=( 8 12 32 )
# minibatch_lst=( "minibatch" )
# method_lst=( "gat-saintrw-10k" ) # "gat-saintrw-5k" )
# gat_heads_lst=( 4 8 )
# lr_lst=( 0.1 0.01 0.001 )

# # redoing gcnjk
# dataset_lst=( "ogbn-arxiv" )
# num_layers_lst=( 2 )
# hidden_channels_lst=( 128 )
# minibatch_lst=( "minibatch" )
# method_lst=( "gcnjk-saintrw-10k" )

# # redoing gat
# dataset_lst=( "ogbn-arxiv" )
# num_layers_lst=( 2 )
# hidden_channels_lst=( 32 )
# minibatch_lst=( "minibatch" )
# method_lst=( "gat-saintrw-10k" )
# gat_heads_lst=( 8 )
# lr_lst=( 0.01 0.001 )

# # redoing mixhop
# dataset_lst=( "ogbn-arxiv" )
# num_layers_lst=( 2 )
# hidden_channels_lst=( 128 )
# minibatch_lst=( "minibatch" )
# method_lst=( "mixhop-saintrw-10k" )

# try mixhop-cluster to steelman the heteromethods
dataset_lst=( "snap-patents" )
num_layers_lst=( 2 4 )
hidden_channels_lst=( 128 )
minibatch_lst=( "minibatch" )
method_lst=( "mixhop-cluster-200-5" "mixhop-cluster-200-1" )

i_d=$(( (${SLURM_ARRAY_TASK_ID} - 1)  % ${#dataset_lst[@]} ))
i_nl=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} % ${#num_layers_lst[@]} ))
i_hc=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} % ${#hidden_channels_lst[@]} ))
i_mb=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} / ${#hidden_channels_lst[@]} % ${#minibatch_lst[@]}))
i_me=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} / ${#hidden_channels_lst[@]} / ${#minibatch_lst[@]} % ${#method_lst[@]}))
i_gh=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} / ${#hidden_channels_lst[@]} / ${#minibatch_lst[@]} / ${#method_lst[@]} % ${#gat_heads_lst[@]}))
i_lr=$(( (${SLURM_ARRAY_TASK_ID} - 1)  / ${#dataset_lst[@]} / ${#num_layers_lst[@]} / ${#hidden_channels_lst[@]} / ${#minibatch_lst[@]} / ${#method_lst[@]} / ${#gat_heads_lst[@]} % ${#lr_lst[@]}))



# runexp    gpu   dataset                  num_layers                   hidden_channels                    minibatch                    method                      train_prop     valid_prop   gat_heads                   lr
runexp      0     ${dataset_lst[${i_d}]}   ${num_layers_lst[${i_nl}]}   ${hidden_channels_lst[${i_hc}]}    ${minibatch_lst[${i_mb}]}    ${method_lst[${i_me}]}      0.5           0.25          ${gat_heads_lst[${i_gh}]}   ${lr_lst[${i_lr}]}


# runexp     gpu    dataset        num_layers    hidden_channels   minibatch    method              train_prop     valid_prop
# if [ ${SLURM_ARRAY_TASK_ID} = 1 ]; then
#     runexp    0     arxiv-year          4           256             minibatch    gcnjk-saintrw-10k    0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 2 ]; then
#     runexp    0     arxiv-year          4           256             minibatch    gcnjk-saintrw-10k    0.2           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 3 ]; then
#     runexp    0     arxiv-year          1           256             minibatch    linkx                0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 4 ]; then
#     runexp    0     arxiv-year          1           256             minibatch    linkx                0.2           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 5 ]; then
#     runexp    0     arxiv-year          4           128             minibatch    mixhop-saintrw-10k   0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 6 ]; then
#     runexp    0     arxiv-year          4           128             minibatch    mixhop-saintrw-10k   0.2           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 7 ]; then
#     runexp    0     snap-patents        2           256             minibatch    gcnjk-saintrw-10k    0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 8 ]; then
#     runexp    0     snap-patents        2           256             minibatch    gcnjk-saintrw-10k    0.2           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 9 ]; then
#     runexp    0     snap-patents        1           16              minibatch    linkx                0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 10 ]; then
#     runexp    0     snap-patents        1           16              minibatch    linkx                0.2           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 11 ]; then
#     runexp    0     snap-patents        2           128             minibatch    mixhop-saintrw-10k   0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 12 ]; then
#     runexp    0     snap-patents        2           128             minibatch    mixhop-saintrw-10k   0.2           0.25
# fi


# if [ ${SLURM_ARRAY_TASK_ID} = 1 ]; then
#     runexp    0     arxiv-year          2           32             minibatch    gat-saintrw-10k    0.1           0.25        8         0.01
# elif [ ${SLURM_ARRAY_TASK_ID} = 2 ]; then
#     runexp    0     arxiv-year          2           32             minibatch    gat-saintrw-10k    0.2           0.25        8         0.01
# elif [ ${SLURM_ARRAY_TASK_ID} = 3 ]; then
#     runexp    0     snap-patents        2           32             minibatch    gat-saintrw-10k    0.1           0.25        8         0.01
# elif [ ${SLURM_ARRAY_TASK_ID} = 4 ]; then
#     runexp    0     snap-patents        2           32             minibatch    gat-saintrw-10k    0.2           0.25        8         0.01
# fi


# if [ ${SLURM_ARRAY_TASK_ID} = 1 ]; then
#     runexp    0     snap-patents          2           128             minibatch    mixhop-cluster-200-5    0.1           0.25
# elif [ ${SLURM_ARRAY_TASK_ID} = 2 ]; then
#     runexp    0     snap-patents          2           128             minibatch    mixhop-cluster-200-5    0.2           0.25
# fi