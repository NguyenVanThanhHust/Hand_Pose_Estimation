#!/bin/bash
python generate_mask.py \
    --input_folder ../../Datasets/CMU_Hand_Dataset/hand_labels/manual_train/ \
    --output_folder ../../Datasets/CMU_Hand_Dataset/hand_labels/manual_train_processed/ 

python generate_mask.py \
    --input_folder ../../Datasets/CMU_Hand_Dataset/hand_labels/manual_test/ \
    --output_folder ../../Datasets/CMU_Hand_Dataset/hand_labels/manual_test_processed/

python generate_mask.py \
    --input_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth1/ \
    --output_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth1_processed/

python generate_mask.py \
    --input_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth2/ \
    --output_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth2_processed/

python generate_mask.py \
    --input_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth3/ \
    --output_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth3_processed/

python generate_mask.py \
    --input_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth4/ \
    --output_folder ../../Datasets/CMU_Hand_Dataset/hand_labels_synth/synth4_processed/
