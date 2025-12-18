#!/bin/bash

COPHY="/Users/alexst-aubin/coding_projects/UBC/Fall2025/cophy_224"
DERENDERING_DIR="./ckpts/derendering"
PREXTRACTED_OBJ="./extracted_obj_visu_prop"
LOG_DIR="./log_dir/cophy_alex"

export PYTORCH_ENABLE_MPS_FALLBACK=1

# -------------------------------
echo "Setting ulimit..."
ulimit -n 10240
# -------------------------------
echo "Running Experiment 1..."
python -m cf_learning.main \
	--dataset_dir "$COPHY/ballsCF" \
    --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
	--log_dir "$LOG_DIR/ballsCF/trans/35" \
	--preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
	--dataset_name balls \
	--model cophynet --num_objects 3 --type normal \
	--batch_size 64 --workers 4 --epochs 10 \
	--encoder_type transformer \
	--w_stab 1.0 --w_pose 2.0 --lr 3e-4 \
	--weight_decay 1e-4 \
	--seed 36

echo "Running Experiment 2..."
python -m cf_learning.main \
	--dataset_dir "$COPHY/ballsCF" \
    --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
	--log_dir "$LOG_DIR/ballsCF/trans/36" \
	--preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
	--dataset_name balls \
	--model cophynet --num_objects 3 --type normal \
	--batch_size 64 --workers 4 --epochs 10 \
	--encoder_type transformer \
	--w_stab 1.0 --w_pose 2.0 --lr 3e-4 \
	--weight_decay 1e-4 \
	--seed 37
echo "Running Experiment 3..."
python -m cf_learning.main \
	--dataset_dir "$COPHY/ballsCF" \
    --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
	--log_dir "$LOG_DIR/ballsCF/trans/37" \
	--preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
	--dataset_name balls \
	--model cophynet --num_objects 3 --type normal \
	--batch_size 64 --workers 4 --epochs 10 \
	--encoder_type transformer \
	--w_stab 1.0 --w_pose 2.0 --lr 3e-4 \
	--weight_decay 1e-4 \
	--seed 38
echo "Running Experiment 4..."
python -m cf_learning.main \
	--dataset_dir "$COPHY/ballsCF" \
    --derendering_ckpt "$DERENDERING_DIR/ballsCF/model_state_dict.pt" \
	--log_dir "$LOG_DIR/ballsCF/trans/38" \
	--preextracted_obj_vis_prop_dir "$PREXTRACTED_OBJ/ballsCF" \
	--dataset_name balls \
	--model cophynet --num_objects 3 --type normal \
	--batch_size 64 --workers 4 --epochs 10 \
	--encoder_type transformer \
	--w_stab 1.0 --w_pose 2.0 --lr 3e-4 \
	--weight_decay 1e-4 \
	--seed 39