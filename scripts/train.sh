#!/bin/bash

dpath="${TMPDIR}/datasets/rsoc"

echo "The data path is set to: $dpath"

python -u train.py \
    --net UDG \
    --gpu 0 \
    --dataset_root $dpath \
    --dataset ship \
    --task 1 \
    --epoch 400 \
    --batch_size 1 \
    --drop_stages 1 2 3 4 \
    --ptb_stages 1 2 3 4
