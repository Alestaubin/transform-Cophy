#!/bin/bash

# --- BASH SYNTAX CORRECTIONS ---
# Use name=value (no spaces)
COPHY="/Users/alexst-aubin/coding_projects/UBC/Fall2025/cophy_224"
DERENDERING_DIR="./ckpts/derendering"
PREXTRACTED_OBJ="./extracted_obj_visu_prop"
LOG_DIR="./log_dir/cophy_alex"

# This one was already correct
export PYTORCH_ENABLE_MPS_FALLBACK=1

# -------------------------------

# echo "Running Experiment 1..."
# python -m cf_learning.main \
#         --dataset_dir "$COPHY/ballsCF" \
#         --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
#         --log_dir "$LOG_DIR/ballsCF/trans/1" \
#         --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
#         --dataset_name balls \
#         --model cophynet --num_objects 3 --type normal \
#         --batch_size 32 --workers 4 --epochs 10 --seed 0 --lr 1e-3 --encoder_type transformer \
#         --w_stab 1.0 --w_pose 1.0

echo "Running Experiment 2..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/2" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 0 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 2.0

echo "Running Experiment 3..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/3" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 0 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 5.0

echo "Running Experiment 4..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/4" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 0 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 8.0