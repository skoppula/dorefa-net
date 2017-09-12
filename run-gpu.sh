gpu=3
model="fc"
state_size=256
n_layers=4
bit_a=4
bit_w=4
data_dir="/data/sls/scratch/skoppula/kaldi-rsr/numpy/"
out_dir="/data/sls/scratch/skoppula/quantized-net/dorefa-net/train_logs_model${model}_nl${n_layers}_ss${state_size}_bita${bit_a}_bitw${bit_w}"
shuffle_q=5000
num_prefetch_threads=4

cmd="python rsr-dorefa.py --gpu=$gpu --data=$data_dir --shuffle_queue_buffer_size=$shuffle_q --bit_w=$bit_w --bit_a=$bit_a --output=$out_dir --num_prefetch_threads=$num_prefetch_threads"
echo $cmd
# eval $cmd

echo "nohup ./run-gpu.sh > train.out 2> train.err < /dev/null &"
