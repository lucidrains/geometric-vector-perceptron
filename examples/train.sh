#!/usr/bin/env bash

DIR=$(dirname "$0")

export CUDA_VISIBLE_DEVICES=1

export PYTHONPATH=$PYTHONPATH:${DIR}

python "${DIR}"/train_lightning.py \
  --default_root_dir ./experiments/ \
  --gpus "0," \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --train_max_len 128 \
  --eval_max_len 128 \
  --accumulate_grad_batches 1 \
  --num_workers 1 \
  --log_gpu_memory "all" \
  --progress_bar_refresh_rate 1 \
  --max_epochs 10 \
  --weights_summary "top" \
  --depth 4 \
  --cutoffs 5.0 \
  --noise 1.0 \
  --init_lr 1e-3 \
