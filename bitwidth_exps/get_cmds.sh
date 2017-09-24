#!/usr/bin/env bash

bit_w=32
bit_a=32

for counter in {1..9}; do
    bit_w=32
    for counter in {1..9}; do
        cmd="python rsr-bitwidth-exps.py --bit_w=$bit_w --bit_a=$bit_a --output=/data/sls/scratch/skoppula/mfcc-nns/rsr-experiments/dorefa/bitwidth_exps/train_logs_a${bit_a}_w${bit_w}/"
        echo $cmd
        bit_w=$(($bit_w-2))
    done
    bit_a=$(($bit_a-2))
done
