exp=$1
if [ $exp -eq 0 ]; then
    bit_a=32
    bit_w=32
    use_clip="False"
    dropout=1
elif [ $exp -eq 1 ]; then
    bit_a=32
    bit_w=32
    use_clip="False"
    dropout=0.9
elif [ $exp -eq 2 ]; then
    bit_a=32
    bit_w=32
    use_clip="False"
    dropout=0.8
elif [ $exp -eq 3 ]; then
    bit_a=32
    bit_w=32
    use_clip="True"
    dropout=1
elif [ $exp -eq 4 ]; then
    bit_a=31
    bit_w=31
    use_clip="False"
    dropout=1
fi

out_dir="/data/sls/scratch/skoppula/mfcc-nns/rsr-experiments/dorefa/train_logs_do${dropout}_bita${bit_a}_bitw${bit_w}_clip${use_clip}"
echo "python rsr-dorefa.py --bit_w=$bit_w --bit_a=$bit_a --output=$out_dir --use_clip=$use_clip --dropout=$dropout"
