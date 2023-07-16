#!/bin/bash

dpath="${TMPDIR}/datasets/rsoc"

echo "The data path is set to: $dpath"

python -u test.py \
    --net UDG \
    --gpu 0 \
    --dataset_root $dpath \
    --dataset ship \
    --task 1 \
    --drop_stages 1 2 3 4 \
    --ptb_stages 1 2 3 4

for num in 1 2 3 4 5 6 7 8 9 10; do
    python -u test.py \
        --net UDG \
        --gpu 0 \
        --dataset_root $dpath \
        --dataset ship \
        --task 1 \
        --drop_stages 1 2 3 4 \
        --ptb_stages 1 2 3 4 \
        --mc_drop \
        --n_forward $num
done
