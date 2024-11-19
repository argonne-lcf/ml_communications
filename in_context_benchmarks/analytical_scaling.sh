SCRIPT_DIR=$(dirname $BASH_SOURCE[0] | xargs realpath)
cd $SCRIPT_DIR

python analytical_scaling.py \
    --batch_size 1 \
    --seq_length 128_000 \
    --hidden_size 8192 \
    --ffn_hidden_size 28672 \
    --num_layers 80 \
    --parallelism 16 \