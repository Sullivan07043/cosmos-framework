# SPDX-License-Identifier: OpenMDW-1.1
"""``action_policy_robocasa_nano`` — Cosmos3-Nano RoboCasa action-policy SFT recipe.

Port of ``action_policy_libero_nano`` to the RoboCasa-Cosmos-Policy demos
(24 kitchen tasks, 1,199 success episodes) converted to LeRobot v3.
Reuses ``get_action_libero_sft_dataset`` / ``LIBEROLeRobotDataset`` unchanged —
the converted data has the identical feature layout (7D frame-wise-relative
[dpos, drot_axisangle, gripper] actions, third-person + wrist concat_view).
Only the data root and the normalizer stats differ.

Env: ``ROBOCASA_ROOT`` must point at the converted LeRobot dir
(contains ``meta/info.json``). Stats file
``robocasa_native_frame_wise_relative_rot6d.json`` must be installed in the
``normalizer_stats`` dir (see install_ports.sh).

Domain note: embodiment stays ``libero`` (domain id 5, Franka + parallel
gripper). The public Cosmos3-Nano base ships no trained action heads — they
init fresh either way — so the id only needs train/eval consistency.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_robocasa_nano_model_config() -> dict:
    """Same model tweaks as the LIBERO recipe: capped packed tokens, selective
    activation checkpointing, fresh diffusion-expert init, 10x vision
    flow-matching loss. ``encode_exact_durations`` matches the Cosmos3-Nano base."""
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]  # match Cosmos3 base (do NOT reduce)
    return cfg


action_policy_robocasa_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss — inherited from the LIBERO recipe).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_robocasa",
            group="action_sft",
            name="action_policy_robocasa_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_robocasa_nano_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke default; real run sets via TOML
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke default; real run sets via TOML
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke default; real run sets via TOML
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema and the action heads so they init fresh from the base
            # (the public Cosmos3-Nano base has no trained action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_robocasa",
            max_samples_per_batch=128,  # H200 bound from the LIBERO recipe; A100-40GB will need less (P1 smoke)
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    robocasa=dict(
                        ratio=1,
                        dataset=L(get_action_libero_sft_dataset)(
                            # Converted RoboCasa-Cosmos-Policy LeRobot dir
                            # (convert_robocasa_to_lerobot.py output).
                            root="${oc.env:ROBOCASA_ROOT}",
                            fps=20,  # metadata only (FPS-agnostic loader reads native fps)
                            chunk_length=16,
                            image_size=256,  # concat_view -> 256x512
                            mode="wam",
                            camera_mode="concat_view",
                            action_space="frame_wise_relative",
                            rotation_space="6d",
                            pose_coordinate_frame="native",
                            action_normalization="quantile_rot",
                            action_stats_path="robocasa_native_frame_wise_relative_rot6d.json",
                            val_ratio=0.01,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            resolution=None,
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=True,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_robocasa_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
