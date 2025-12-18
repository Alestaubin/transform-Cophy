#!/bin/bash

# --- BASH SYNTAX CORRECTIONS ---
# Use name=value (no spaces)
COPHY="/Users/alexst-aubin/coding_projects/UBC/Fall2025/cophy_224"
DERENDERING_DIR="./ckpts/derendering"
PREXTRACTED_OBJ="./extracted_obj_visu_prop"
LOG_DIR="./log_dir/cophy_alex"

# This one was already correct
export PYTORCH_ENABLE_MPS_FALLBACK=1
ulimit -n 10240
# -------------------------------

echo "Running Experiment 1..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/rnn/20" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 1 --lr 1e-3 --encoder_type rnn \
        --w_stab 1.0 --w_pose 10.0
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/20" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 1 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 2.0

echo "Running Experiment 2..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/rnn/21" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 2 --lr 1e-3 --encoder_type rnn \
        --w_stab 1.0 --w_pose 10.0
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/21" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 2 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 2.0

echo "Running Experiment 3..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/rnn/22" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 3 --lr 1e-3 --encoder_type rnn \
        --w_stab 1.0 --w_pose 10.0
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/22" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 3 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 2.0

echo "Running Experiment 4..."
python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/rnn/23" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 4 --lr 1e-3 --encoder_type rnn \
        --w_stab 1.0 --w_pose 10.0

python -m cf_learning.main \
        --dataset_dir "$COPHY/ballsCF" \
        --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
        --log_dir "$LOG_DIR/ballsCF/trans/24" \
        --preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
        --dataset_name balls \
        --model cophynet --num_objects 3 --type normal \
        --batch_size 32 --workers 4 --epochs 10 --seed 66 --lr 1e-3 --encoder_type transformer \
        --w_stab 1.0 --w_pose 2.0

