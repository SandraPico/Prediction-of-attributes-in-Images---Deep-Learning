#!/bin/sh

DATASET_DIR_SMILE=/Volumes/Marine_WD_Elements/Dataset_DL/dataset_smile
DATASET_DIR_SMILE_CROPPED=/Volumes/Marine_WD_Elements/Dataset_DL/dataset_smile_cropped
DATASET_DIR_GENDER=/Volumes/Marine_WD_Elements/Dataset_DL/dataset_gender
DATASET_DIR_GENDER_CROPPED=/Volumes/Marine_WD_Elements/Dataset_DL/dataset_gender_cropped

TRAIN_DIR=/Volumes/Marine_WD_Elements/DL_PRETRAINED


/Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_SMILE} \
--output_graph ${TRAIN_DIR}/SMILE_inceptionv3_0_01/output_graph.pb \
--intermediate_output_graphs_dir ${TRAIN_DIR}/SMILE_inceptionv3_0_01/intermediate_graph \
--output_labels ${TRAIN_DIR}/SMILE_inceptionv3_0_01/output_labels.txt \
--summaries_dir ${TRAIN_DIR}/SMILE_inceptionv3_0_01/retrain_logs \
--bottleneck_dir ${TRAIN_DIR}/SMILE_inceptionv3_0_01/bottleneck \
--saved_model_dir${TRAIN_DIR}/SMILE_inceptionv3_0_01/model \
--print_misclassified_test_images



/Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_SMILE_CROPPED} \
--output_graph ${TRAIN_DIR}/SMILE_CROPPED_inceptionv3_0_01/output_graph.pb \
--intermediate_output_graphs_dir ${TRAIN_DIR}/SMILE_CROPPED_inceptionv3_0_01/intermediate_graph \
--output_labels ${TRAIN_DIR}/SMILE_CROPPED_inceptionv3_0_01/output_labels.txt \
--summaries_dir ${TRAIN_DIR}/SMILE_CROPPED_inceptionv3_0_01/retrain_logs \
--bottleneck_dir ${TRAIN_DIR}/SMILE_CROPPED_inceptionv3_0_01/bottleneck \
--saved_model_dir${TRAIN_DIR}/SMILE_CROPPED_inceptionv3_0_01/model \
--print_misclassified_test_images


/Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_GENDER} \
--output_graph ${TRAIN_DIR}/GENDER_inceptionv3_0_01/output_graph.pb \
--intermediate_output_graphs_dir ${TRAIN_DIR}/GENDER_inceptionv3_0_01/intermediate_graph \
--output_labels ${TRAIN_DIR}/GENDER_inceptionv3_0_01/output_labels.txt \
--summaries_dir ${TRAIN_DIR}/GENDER_inceptionv3_0_01/retrain_logs \
--bottleneck_dir ${TRAIN_DIR}/GENDER_inceptionv3_0_01/bottleneck \
--saved_model_dir${TRAIN_DIR}/GENDER_inceptionv3_0_01/model \
--print_misclassified_test_images

/Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_GENDER_CROPPED} \
--output_graph ${TRAIN_DIR}/GENDER_CROPPED_inceptionv3_0_01/output_graph.pb \
--intermediate_output_graphs_dir ${TRAIN_DIR}/GENDER_CROPPED_inceptionv3_0_01/intermediate_graph \
--output_labels ${TRAIN_DIR}/GENDER_CROPPED_inceptionv3_0_01/output_labels.txt \
--summaries_dir ${TRAIN_DIR}/GENDER_CROPPED_inceptionv3_0_01/retrain_logs \
--bottleneck_dir ${TRAIN_DIR}/GENDER_CROPPED_inceptionv3_0_01/bottleneck \
--saved_model_dir${TRAIN_DIR}/GENDER_CROPPED_inceptionv3_0_01/model \
--print_misclassified_test_images






# #Complementary NASNET
# # NASNET default

# /Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_SMILE} \
# --output_graph ${TRAIN_DIR}/SMILE_nasnet_0_01/output_graph.pb \
# --intermediate_output_graphs_dir ${TRAIN_DIR}/SMILE_nasnet_0_01/intermediate_graph \
# --output_labels ${TRAIN_DIR}/SMILE_nasnet_0_01/output_labels.txt \
# --summaries_dir ${TRAIN_DIR}/SMILE_nasnet_0_01/retrain_logs \
# --bottleneck_dir ${TRAIN_DIR}/SMILE_nasnet_0_01/bottleneck \
# --saved_model_dir ${TRAIN_DIR}/SMILE_nasnet_0_01/model \
# --tfhub_module https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1 \
# --print_misclassified_test_images



# # NASNET Learning rate changed
# /Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_SMILE} \
# --output_graph ${TRAIN_DIR}/SMILE_nasnet_0_001/output_graph.pb \
# --intermediate_output_graphs_dir ${TRAIN_DIR}/SMILE_nasnet_0_001/intermediate_graph \
# --output_labels ${TRAIN_DIR}/SMILE_nasnet_0_001/output_labels.txt \
# --summaries_dir ${TRAIN_DIR}/SMILE_nasnet_0_001/retrain_logs \
# --bottleneck_dir ${TRAIN_DIR}/SMILE_nasnet_0_001/bottleneck \
# --saved_model_dir ${TRAIN_DIR}/SMILE_nasnet_0_001/model \
# --tfhub_module https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1 \
# --learning_rate 0.001

# # NASNET Learning rate changed
# /Users/Marine/anaconda2/envs/tensorflow/bin/python retrain.py --image_dir ${DATASET_DIR_SMILE} \
# --output_graph ${TRAIN_DIR}/SMILE_nasnet_0_05/output_graph.pb \
# --intermediate_output_graphs_dir ${TRAIN_DIR}/SMILE_nasnet_0_05/intermediate_graph \
# --output_labels ${TRAIN_DIR}/SMILE_nasnet_0_05/output_labels.txt \
# --summaries_dir ${TRAIN_DIR}/SMILE_nasnet_0_05/retrain_logs \
# --bottleneck_dir ${TRAIN_DIR}/SMILE_nasnet_0_05/bottleneck \
# --saved_model_dir ${TRAIN_DIR}/SMILE_nasnet_0_05/model \
# --tfhub_module https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1 \
# --learning_rate 0.05







 


/Users/Marine/anaconda2/envs/tensorflow/bin/tensorboard --logdir /Volumes/Marine_WD_Elements/DL_PRETRAINED/tmp2/retrain_logs

