#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

# Where the scripts are
SCRIPT_DIR=/cfs/klemming/nobackup/c/collery/DL/slim

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/cfs/klemming/nobackup/c/collery/DL/tmp/checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/cfs/klemming/nobackup/c/collery/DL/result/GENDER_CROPPED/train
EVAL_DIR=/cfs/klemming/nobackup/c/collery/DL/result/GENDER_CROPPED/eval


# Where the dataset is saved to.
DATASET_DIR=/cfs/klemming/nobackup/c/collery/DL/dataset/gender_cropped

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

# Download the dataset
python ${SCRIPT_DIR}/download_and_convert_data_MTFL.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}



# #Fine-tune only the new layers for 1000 steps.
python ${SCRIPT_DIR}/train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=100 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation on validation set batch size = 100
python ${SCRIPT_DIR}/eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${EVAL_DIR}/val \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3


# Fine-tune all the new layers for 500 steps.
python ${SCRIPT_DIR}/train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=100 \
  --learning_rate=0.001 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


# Run evaluation on validation set batch size = 100
python ${SCRIPT_DIR}/eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${EVAL_DIR}/all/val \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3


# Run evaluation on test set batch size = 100
python ${SCRIPT_DIR}/eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${EVAL_DIR}/all/test \
  --dataset_name=flowers \
  --dataset_split_name=testing \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3