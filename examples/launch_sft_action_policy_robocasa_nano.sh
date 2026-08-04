#!/usr/bin/env bash
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_robocasa_nano — Cosmos3-Nano RoboCasa
# action-policy SFT (HSDP, full SFT of generation + action heads). Drives
# cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_robocasa_nano.toml.
#
# Required env vars:
#   ROBOCASA_ROOT         converted RoboCasa LeRobot dir (contains meta/info.json)
# Optional env vars:
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano (DCP dir)
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   OUTPUT_ROOT           default: outputs/train
#   NNODES/NODE_RANK/MASTER_ADDR   multi-node topology (unset = single node)
#   EXTRA_TAIL_OVERRIDES  Hydra overrides, e.g. "trainer.max_iter=5"
#
# Usage (HSDP 2x8):
#   node0: NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> ROBOCASA_ROOT=<dir> bash <this script>
#   node1: NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-ip> ROBOCASA_ROOT=<dir> bash <this script>
# Single-node smoke (8 GPUs):
#   ROBOCASA_ROOT=<dir> EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 model.parallelism.data_parallel_replicate_degree=1" bash <this script>

TOML_FILE="examples/toml/sft_config/action_policy_robocasa_nano.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

export ROBOCASA_ROOT="${ROBOCASA_ROOT:-}"

EXTRA_DATASET_CHECK='[[ -f "$ROBOCASA_ROOT/meta/info.json" ]] || { echo "ERROR: ROBOCASA_ROOT must be a converted LeRobot dir containing meta/info.json (got: '\''$ROBOCASA_ROOT'\''). Run convert_robocasa_to_lerobot.py first." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
