python main_scalable.py \
    --dataset ogbn-products \
    --method linkx \
    --num_layers 1 \
    --hidden_channels 32 \
    --display_step 25 \
    --runs 5 \
    --train_batch row \
    --num_parts 10

python main_scalable.py \
    --dataset ogbn-products \
    --method mixhop --hops 2 \
    --num_layers 1 \
    --hidden_channels 64 \
    --display_step 25 \
    --runs 5 \
    --train_batch graphsaint-rw \
    --batch_size 5000 \
    --saint_num_steps 5 \
    --test_num_parts 10


#### Products runs it just needed a larger gpu and the code change

waiting...

#### now to try the GCN script

python main_scalable.py --dataset arxiv-year \
    --method gcnjk --jk_type cat --num_layers 1 --hidden_channels 128 \
    --display_step 25 --runs 5 --train_batch graphsaint-rw --batch_size 10000 \
    --no_mini_batch_test --saint_num_steps 5

### seems to work hope most of the gcnjk runs execute! 5/15 at night


### 5 16 730p ~~nearly~~ have the faction script ready and spinning ... but the test logic might be less than ideal


# Maybe the first step before pulling in separate code is to try to get the GAT from the fullbatch setting to run in minibatch setting...


python main.py \
    --dataset arxiv-year \
    --method gatjk \
    --num_layers 2 \
    --hidden_channels 12 \
    --lr 0.01 \
    --gat_heads 4 \
    --directed --display_step 25 \
    --runs 5 
 
 works ^^
hmmm...

python main_scalable.py \
    --dataset arxiv-year \
    --method gat \
    --num_layers 2 \
    --hidden_channels 12 \
    --lr 0.01 \
    --gat_heads 4 \
    --directed --display_step 25 \
    --runs 5 \
    --train_batch graphsaint-rw --batch_size 10000 \
    --no_mini_batch_test --saint_num_steps 5


Debugging... 
expname=${method}-${minibatch}-${dataset}${sub_dataset}-${num_layers}-${hidden_channels}-${train_prop}-${valid_prop}-${gat_heads}-${lr}

gat-saintrw-10k-minibatch-arxiv-year-2-4-0.5-0.25-2-0.01.log

python main_scalable.py --dataset arxiv-year \
    --method gat --num_layers 2 --hidden_channels 4 \
    --lr 0.01 --gat_heads 2 --directed --display_step 25 --runs 5 \
    --train_batch graphsaint-rw --batch_size 10000 \
    --no_mini_batch_test --saint_num_steps 5



################################

debugging the ogbn-arxiv dataloading



python main_scalable.py \
    --dataset ogbn-arxiv \
    --method gcnjk \
    --jk_type cat \
    --num_layers 2 \
    --hidden_channels 128 \
    --display_step 25 \
    --runs 5 \
    --train_batch graphsaint-rw \
    --batch_size 10000 \
    --no_mini_batch_test \
    --saint_num_steps 5 \
    --test_num_parts 10


python main_scalable.py \
    --dataset ogbn-arxiv \
    --method gat \
    --num_layers 2 \
    --hidden_channels 32 \
    --lr 0.001 \
    --gat_heads 8 \
    --display_step 25 \
    --runs 5 \
    --train_batch graphsaint-rw \
    --batch_size 10000 \
    --saint_num_steps 5 \
    --no_mini_batch_test


python main_scalable.py \
    --dataset ogbn-arxiv \
    --method mixhop \
    --hops 2 \
    --num_layers 2 \
    --hidden_channels 128 \
    --display_step 25 \
    --runs 5 \
    --train_batch graphsaint-rw \
    --batch_size 10000 \
    --saint_num_steps 5 \
    --no_mini_batch_test

odd one: 

Namespace(SGD=False, adam=False, batch_size=10000, cached=False, cluster_batch_size=1, cpu=False, dataset='ogbn-arxiv', directed=False, display_step=25, dropout=0.5, epochs=500, gat_heads=8, gcn2_alpha=0.1, gpr_alpha=0.1, hidden_channels=128, hops=1, inner_activation=False, inner_dropout=False, jk_type='cat', link_init_layers_A=1, link_init_layers_X=1, lp_alpha=0.1, lr=0.01, method='gcnjk', no_bn=False, no_mini_batch_test=False, num_layers=2, num_mlp_layers=1, num_parts=100, print_prop=False, rand_split=False, rocauc=False, runs=5, saint_num_steps=5, sampling=False, sub_dataset='', test_num_parts=10, theta=0.5, train_batch='graphsaint-rw', train_prop=0.5, valid_prop=0.25, weight_decay=0.001)

working one: 

Namespace(SGD=False, adam=False, batch_size=10000, cached=False, cluster_batch_size=1, cpu=False, dataset='ogbn-arxiv', directed=False, display_step=25, dropout=0.5, epochs=500, gat_heads=8, gcn2_alpha=0.1, gpr_alpha=0.1, hidden_channels=128, hops=1, inner_activation=False, inner_dropout=False, jk_type='cat', link_init_layers_A=1, link_init_layers_X=1, lp_alpha=0.1, lr=0.01, method='gcnjk', no_bn=False, no_mini_batch_test=True, num_layers=2, num_mlp_layers=1, num_parts=100, print_prop=False, rand_split=False, rocauc=False, runs=5, saint_num_steps=5, sampling=False, sub_dataset='', test_num_parts=10, theta=0.5, train_batch='graphsaint-rw', train_prop=0.5, valid_prop=0.25, weight_decay=0.001)

no_mini_batch_test=True was the key!!



### Mixhop cluster attempt

python main_scalable.py \
    --dataset snap-patents \
    --method mixhop \
    --num_layers 2 \
    --hidden_channels 128 \
    --display_step 25 \
    --runs 1 \
    --hops 2 \
    --train_batch cluster \
    --num_parts 200 \
    --cluster_batch_size 5
