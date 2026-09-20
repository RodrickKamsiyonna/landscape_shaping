import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
import custom_resolvers  # noqa: F401

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

SAVE_EVERY_ITERS = 500


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()

        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("checkpoints/")[-1]
        model_name += f"_f{cfg.frameskip}_h{cfg.num_hist}_p{cfg.num_pred}"

        # --------------------------------------------------------------
        # Distributed initialization
        # --------------------------------------------------------------
        if HydraConfig.get().mode == RunMode.MULTIRUN:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=5),
            )
            dist.barrier()

        mixed_precision = cfg.training.get("mixed_precision", "no")

        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=mixed_precision,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=bool(
                        cfg.get("has_decoder", False)
                    )
                )
            ],
        )

        self.device = self.accelerator.device
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = (
            cfg.training.num_reconstruct_samples
        )

        # cfg.training.epochs is the number of epochs requested for a
        # fresh run. We preserve the original intended final epoch in
        # self.target_epoch so a resumed job does NOT accidentally run
        # another cfg.training.epochs epochs.
        self.total_epochs = int(cfg.training.epochs)

        self.epoch = 0
        self.global_step = 0

        # --------------------------------------------------------------
        # Resume state
        # --------------------------------------------------------------
        #
        # resume_phase:
        #
        #   "train"
        #       checkpoint was taken inside training.
        #
        #   "train_complete"
        #       training for this epoch is complete, validation has not
        #       yet completed.
        #
        #   "epoch_complete"
        #       the whole epoch is complete; resume from next epoch.
        #
        self.resume_phase = "epoch_complete"

        # Number of TRAIN batches already completed in resume epoch.
        # This is the index of the NEXT batch to execute.
        self.resume_iteration = 0

        # Absolute final epoch for this run.
        self.target_epoch = self.total_epochs

        # Partial accumulated metrics for an interrupted epoch.
        self.epoch_log = OrderedDict()

        # Loaded RNG state is restored only after model/optimizer
        # initialization so initialization itself cannot corrupt the
        # restored training RNG state.
        self._loaded_rng_states = None

        self.save_every_iters = SAVE_EVERY_ITERS

        log.info(
            f"Accelerate mixed precision: {mixed_precision}"
        )
        log.info(
            f"rank={self.accelerator.local_process_index} "
            f"device={self.device}"
        )
        log.info(
            f"Mid-epoch checkpoint every "
            f"{self.save_every_iters} global steps"
        )

        self.decoder_start_epoch = int(
            cfg.training.get("decoder_start_epoch", 1)
        )
        self.decoder_start_epoch = max(
            1,
            self.decoder_start_epoch,
        )

        assert (
            cfg.training.batch_size
            % self.accelerator.num_processes
            == 0
        ), (
            "Batch size must be divisible by number of processes. "
            f"batch_size={cfg.training.batch_size}, "
            f"num_processes={self.accelerator.num_processes}"
        )

        OmegaConf.set_struct(cfg, False)

        cfg.effective_batch_size = cfg.training.batch_size

        cfg.gpu_batch_size = (
            cfg.training.batch_size
            // self.accelerator.num_processes
        )

        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()

        # --------------------------------------------------------------
        # WandB
        # --------------------------------------------------------------
        if self.accelerator.is_main_process:
            wandb_run_id = None

            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg.get("wandb_run_id")

                if wandb_run_id is not None:
                    log.info(
                        f"Resuming WandB run {wandb_run_id}"
                    )

            wandb_dict = OmegaConf.to_container(
                cfg,
                resolve=True,
            )

            self.wandb_run = wandb.init(
                project=f"temporal_straightening_{cfg.env.name}",
                config=wandb_dict,
                id=wandb_run_id,
                resume="allow",
            )

            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)

            self.wandb_run.name = model_name

            with open(
                os.path.join(os.getcwd(), "hydra.yaml"),
                "w",
            ) as f:
                f.write(
                    OmegaConf.to_yaml(
                        cfg,
                        resolve=True,
                    )
                )

        # --------------------------------------------------------------
        # Seed before dataset/model construction
        # --------------------------------------------------------------
        seed(cfg.training.seed)

        log.info(
            f"Loading dataset from "
            f"{cfg.env.dataset.data_path} ..."
        )

        self.datasets, traj_dsets = hydra.utils.call(
            cfg.env.dataset,
            num_hist=cfg.num_hist,
            num_pred=cfg.num_pred,
            frameskip=cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        # --------------------------------------------------------------
        # DataLoaders
        #
        # IMPORTANT:
        # shuffle=False is required for deterministic mid-epoch
        # position reconstruction by batch index.
        # --------------------------------------------------------------
        self.dataloaders = {
            split: torch.utils.data.DataLoader(
                self.datasets[split],
                batch_size=cfg.gpu_batch_size,
                shuffle=False,
                num_workers=cfg.env.num_workers,
                collate_fn=None,
                pin_memory=True,
                persistent_workers=cfg.env.num_workers > 0,
            )
            for split in ["train", "valid"]
        }

        (
            self.dataloaders["train"],
            self.dataloaders["valid"],
        ) = self.accelerator.prepare(
            self.dataloaders["train"],
            self.dataloaders["valid"],
        )

        log.info(
            f"Dataloader batch size per GPU: "
            f"{cfg.gpu_batch_size}"
        )

        self.train_num_batches = len(
            self.dataloaders["train"]
        )

        log.info(
            f"Training batches per epoch: "
            f"{self.train_num_batches}"
        )

        # --------------------------------------------------------------
        # Model references
        # --------------------------------------------------------------
        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None

        self.train_encoder = cfg.model.train_encoder
        self.train_predictor = cfg.model.train_predictor
        self.train_decoder = cfg.model.train_decoder

        # --------------------------------------------------------------
        # Everything below is serialized in our checkpoint.
        # --------------------------------------------------------------
        self._keys_to_save = [
            "epoch",
            "global_step",
            "resume_phase",
            "resume_iteration",
            "target_epoch",
            "epoch_log",
            "train_num_batches",
            "gpu_batch_size",
            "num_processes",
            "encoder_config_seed",
        ]

        if self.train_encoder:
            self._keys_to_save += [
                "encoder",
                "encoder_optimizer",
            ]

        if (
            self.train_predictor
            and cfg.has_predictor
        ):
            self._keys_to_save += [
                "predictor",
                "predictor_optimizer",
                "action_encoder_optimizer",
            ]

        if self.train_decoder:
            self._keys_to_save += [
                "decoder",
                "decoder_optimizer",
            ]

        self._keys_to_save += [
            "action_encoder",
            "proprio_encoder",
            "rng_states",
        ]

        # A harmless value used only to detect an incompatible
        # checkpoint configuration.
        self.encoder_config_seed = int(
            cfg.training.seed
        )

        # --------------------------------------------------------------
        # Initialize models / optimizers.
        #
        # init_models() can load model_latest.pth before constructing
        # missing modules.
        # --------------------------------------------------------------
        self.init_models()
        self.init_optimizers()

        # --------------------------------------------------------------
        # Backward compatibility for old checkpoints
        # --------------------------------------------------------------
        if self.global_step == 0 and self.epoch > 0:
            self.global_step = (
                self.epoch
                * len(self.dataloaders["train"])
            )

            log.warning(
                "Checkpoint did not contain global_step. "
                "Estimated global_step from epoch."
            )

        # Old checkpoints do not have these new resume fields.
        if not hasattr(self, "resume_phase"):
            self.resume_phase = "epoch_complete"

        if not hasattr(self, "resume_iteration"):
            self.resume_iteration = 0

        if not hasattr(self, "target_epoch"):
            self.target_epoch = (
                self.epoch + self.total_epochs
            )

        if not hasattr(self, "epoch_log"):
            self.epoch_log = OrderedDict()

        # --------------------------------------------------------------
        # Restore RNG only AFTER model + optimizer initialization.
        # --------------------------------------------------------------
        self.restore_loaded_rng_state()

    # ------------------------------------------------------------------
    # RNG state
    # ------------------------------------------------------------------

    def _local_rng_state(self):
        """
        Capture all relevant RNG state for the current process.

        This allows random operations to continue from the same point
        after restarting from a checkpoint.
        """
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["cuda"] = [
                x.clone()
                for x in torch.cuda.get_rng_state_all()
            ]
        else:
            state["cuda"] = None

        return state

    def _collect_rng_states(self):
        """
        Collect RNG state from every distributed rank.

        Each process loads the same checkpoint after restart and
        restores the state corresponding to its rank.
        """
        local_state = self._local_rng_state()

        if (
            self.accelerator.num_processes > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
            gathered = [
                None
                for _ in range(
                    self.accelerator.num_processes
                )
            ]

            dist.all_gather_object(
                gathered,
                local_state,
            )

            return gathered

        return [local_state]

    def restore_loaded_rng_state(self):
        """
        Restore the RNG state belonging to this process.
        """
        if self._loaded_rng_states is None:
            return

        states = self._loaded_rng_states

        if not isinstance(states, list):
            log.warning(
                "Checkpoint RNG state has an unexpected format. "
                "Skipping RNG restoration."
            )
            return

        rank = self.accelerator.process_index

        if rank >= len(states):
            log.warning(
                "No RNG state found for rank=%d. "
                "Skipping RNG restoration.",
                rank,
            )
            return

        state = states[rank]

        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(
                state["torch"]
            )

            if (
                torch.cuda.is_available()
                and state.get("cuda") is not None
            ):
                torch.cuda.set_rng_state_all(
                    state["cuda"]
                )

            log.info(
                f"Restored RNG state for rank={rank}"
            )

        except Exception as exc:
            log.warning(
                f"Could not restore RNG state: {exc}"
            )

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _configure_encoder_trainability(self):
        base_model = getattr(
            self.encoder,
            "base_model",
            None,
        )

        if base_model is not None:
            for p in base_model.parameters():
                p.requires_grad = False

            log.info(
                "Encoder base_model frozen."
            )

        else:
            log.info(
                "Encoder has no base_model; "
                "nothing to freeze."
            )

        if not self.train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

            log.info(
                "Encoder fully frozen."
            )
            return

        if base_model is not None:
            for name, p in self.encoder.named_parameters():
                if not name.startswith("base_model."):
                    p.requires_grad = True

            log.info(
                "Encoder backbone frozen; "
                "extra modules trainable."
            )

        else:
            log.info(
                "Encoder fully trainable."
            )

    def _log_trainable_params(
        self,
        module,
        name,
    ):
        if not self.accelerator.is_main_process:
            return

        total = sum(
            p.numel()
            for p in module.parameters()
        )

        trainable = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

        log.info(
            f"[{name}] trainable params: "
            f"{trainable}/{total}"
        )

    def decoder_training_active(self):
        return (
            self.cfg.has_decoder
            and self.train_decoder
            and self.epoch
            >= self.decoder_start_epoch
        )

    def init_models(self):
        model_ckpt = (
            Path(self.cfg.saved_folder)
            / "checkpoints"
            / "model_latest.pth"
        )

        # --------------------------------------------------------------
        # Load checkpoint BEFORE constructing missing models.
        # --------------------------------------------------------------
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)

            log.info(
                f"Loaded checkpoint: {model_ckpt}"
            )

            log.info(
                f"Resume state: "
                f"epoch={self.epoch}, "
                f"phase={self.resume_phase}, "
                f"next_train_batch={self.resume_iteration}, "
                f"global_step={self.global_step}, "
                f"target_epoch={self.target_epoch}"
            )

        # --------------------------------------------------------------
        # Encoder
        # --------------------------------------------------------------
        if self.encoder is None:
            encoder_kwargs = {}

            if (
                hasattr(
                    self.cfg.encoder,
                    "projector_config",
                )
                and self.cfg.encoder.projector_config
                is not None
            ):
                encoder_kwargs["projector_config"] = (
                    hydra.utils.instantiate(
                        self.cfg.encoder.projector_config
                    )
                )

            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
                **encoder_kwargs,
            )

        self._configure_encoder_trainability()

        # --------------------------------------------------------------
        # Proprio encoder
        # --------------------------------------------------------------
        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                self.cfg.proprio_encoder,
                in_chans=self.datasets["train"].proprio_dim,
                emb_dim=self.cfg.proprio_emb_dim,
            )

        # --------------------------------------------------------------
        # Action encoder
        # --------------------------------------------------------------
        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.datasets["train"].action_dim,
                emb_dim=self.cfg.action_emb_dim,
            )

        proprio_emb_dim = (
            self.proprio_encoder.emb_dim
        )
        action_emb_dim = (
            self.action_encoder.emb_dim
        )

        if self.accelerator.is_main_process:
            self.wandb_run.watch(
                self.action_encoder
            )
            self.wandb_run.watch(
                self.proprio_encoder
            )

        # --------------------------------------------------------------
        # Number of visual patches
        # --------------------------------------------------------------
        if self.encoder.latent_ndim == 1:
            num_patches = 1
        else:
            decoder_scale = 16
            num_side_patches = (
                self.cfg.img_size
                // decoder_scale
            )
            num_patches = (
                num_side_patches ** 2
            )

        if self.cfg.concat_dim == 0:
            num_patches += 2

        # --------------------------------------------------------------
        # Predictor
        # --------------------------------------------------------------
        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=(
                        self.encoder.emb_dim
                        + (
                            proprio_emb_dim
                            * self.cfg.num_proprio_repeat
                            + action_emb_dim
                            * self.cfg.num_action_repeat
                        )
                        * self.cfg.concat_dim
                    ),
                )

            if not self.train_predictor:
                for p in self.predictor.parameters():
                    p.requires_grad = False

        # --------------------------------------------------------------
        # Decoder
        # --------------------------------------------------------------
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path,
                        self.cfg.env.decoder_path,
                    )

                    ckpt = torch.load(
                        decoder_path,
                        weights_only=False,
                    )

                    self.decoder = (
                        ckpt["decoder"]
                        if isinstance(
                            ckpt,
                            dict,
                        )
                        else ckpt
                    )

                else:
                    decoder_kwargs = {
                        "emb_dim": self.encoder.emb_dim
                    }

                    if (
                        hasattr(
                            self.cfg.encoder,
                            "projector_config",
                        )
                        and self.cfg.encoder.projector_config
                        is not None
                        and "conv_layers"
                        in self.cfg.encoder.projector_config
                    ):
                        decoder_kwargs[
                            "projector_cfg"
                        ] = (
                            self.cfg.encoder.projector_config
                        )

                    decoder_kwargs["_recursive_"] = False

                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        **decoder_kwargs,
                    )

            if not self.train_decoder:
                for p in self.decoder.parameters():
                    p.requires_grad = False

        # --------------------------------------------------------------
        # Complete world model
        # --------------------------------------------------------------
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            straighten=self.cfg.training.get(
                "straighten",
                False,
            ),
            stop_grad=self.cfg.training.get(
                "stop_grad",
                True,
            ),
            vcreg=self.cfg.training.get(
                "vcreg",
                False,
            ),
            vcreg_std_coeff=self.cfg.training.get(
                "vcreg_std_coeff",
                0,
            ),
            vcreg_cov_coeff=self.cfg.training.get(
                "vcreg_cov_coeff",
                0,
            ),
            vcreg_apply_to=self.cfg.training.get(
                "vcreg_apply_to",
                "enc",
            ),
            landscape_shaping=self.cfg.training.get(
                "landscape_shaping",
                False,
            ),
            eqm_lambda=self.cfg.training.get(
                "eqm_lambda",
                1.0,
            ),
            eqm_weight=self.cfg.training.get(
                "eqm_weight",
                0.5,
            ),
        )

        self._log_trainable_params(
            self.model,
            "model",
        )

        self.ddp_model = self.accelerator.prepare(
            self.model
        )

    def init_optimizers(self):
        # --------------------------------------------------------------
        # Encoder optimizer
        # --------------------------------------------------------------
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )

        self.encoder_optimizer = (
            self.accelerator.prepare(
                self.encoder_optimizer
            )
        )

        if getattr(
            self,
            "_loaded_optim_state",
            None,
        ):
            state = (
                self._loaded_optim_state.get(
                    "encoder_optimizer"
                )
            )

            if state is not None:
                self.encoder_optimizer.load_state_dict(
                    state
                )

        # --------------------------------------------------------------
        # Predictor + action/proprio optimizers
        # --------------------------------------------------------------
        if self.cfg.has_predictor:
            self.predictor_optimizer = (
                torch.optim.AdamW(
                    self.predictor.parameters(),
                    lr=self.cfg.training.predictor_lr,
                )
            )

            self.predictor_optimizer = (
                self.accelerator.prepare(
                    self.predictor_optimizer
                )
            )

            if getattr(
                self,
                "_loaded_optim_state",
                None,
            ):
                state = (
                    self._loaded_optim_state.get(
                        "predictor_optimizer"
                    )
                )

                if state is not None:
                    self.predictor_optimizer.load_state_dict(
                        state
                    )

            self.action_encoder_optimizer = (
                torch.optim.AdamW(
                    itertools.chain(
                        self.action_encoder.parameters(),
                        self.proprio_encoder.parameters(),
                    ),
                    lr=self.cfg.training.action_encoder_lr,
                )
            )

            self.action_encoder_optimizer = (
                self.accelerator.prepare(
                    self.action_encoder_optimizer
                )
            )

            if getattr(
                self,
                "_loaded_optim_state",
                None,
            ):
                state = (
                    self._loaded_optim_state.get(
                        "action_encoder_optimizer"
                    )
                )

                if state is not None:
                    self.action_encoder_optimizer.load_state_dict(
                        state
                    )

        # --------------------------------------------------------------
        # Decoder optimizer
        # --------------------------------------------------------------
        if self.cfg.has_decoder:
            self.decoder_optimizer = (
                torch.optim.Adam(
                    self.decoder.parameters(),
                    lr=self.cfg.training.decoder_lr,
                )
            )

            self.decoder_optimizer = (
                self.accelerator.prepare(
                    self.decoder_optimizer
                )
            )

            if getattr(
                self,
                "_loaded_optim_state",
                None,
            ):
                state = (
                    self._loaded_optim_state.get(
                        "decoder_optimizer"
                    )
                )

                if state is not None:
                    self.decoder_optimizer.load_state_dict(
                        state
                    )

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def _validate_checkpoint_compatibility(
        self,
        ckpt,
    ):
        """
        Make sure a mid-epoch checkpoint is being resumed with the
        same DataLoader layout.

        Since the checkpoint stores a batch index, changing the local
        batch size / number of processes would make that index invalid.
        """
        saved_num_batches = ckpt.get(
            "train_num_batches"
        )
        saved_gpu_batch_size = ckpt.get(
            "gpu_batch_size"
        )
        saved_num_processes = ckpt.get(
            "num_processes"
        )

        # Old checkpoints don't have these fields.
        if saved_num_batches is None:
            return

        if (
            saved_gpu_batch_size is not None
            and int(saved_gpu_batch_size)
            != int(self.cfg.gpu_batch_size)
        ):
            raise RuntimeError(
                "Cannot safely resume checkpoint: "
                f"gpu_batch_size changed from "
                f"{saved_gpu_batch_size} to "
                f"{self.cfg.gpu_batch_size}."
            )

        if (
            saved_num_processes is not None
            and int(saved_num_processes)
            != int(self.accelerator.num_processes)
        ):
            raise RuntimeError(
                "Cannot safely resume checkpoint: "
                f"num_processes changed from "
                f"{saved_num_processes} to "
                f"{self.accelerator.num_processes}."
            )

        if (
            int(saved_num_batches)
            != int(self.train_num_batches)
        ):
            raise RuntimeError(
                "Cannot safely resume checkpoint: "
                f"number of training batches changed "
                f"from {saved_num_batches} to "
                f"{self.train_num_batches}."
            )

    def save_ckpt(
        self,
        phase="epoch_complete",
        iteration=0,
        save_epoch_copy=False,
    ):
        """
        Save a checkpoint.

        phase:
            train
                A checkpoint taken after a completed train batch.
                `iteration` is the NEXT train batch to execute.

            train_complete
                Training for the epoch has completed. Validation
                has not completed.

            epoch_complete
                Entire epoch has completed.

        iteration:
            Number of train batches already completed.
            Example:
                batch 0..499 completed -> iteration=500
                resume starts at batch 500.
        """
        self.accelerator.wait_for_everyone()

        # Update resume state before taking the checkpoint snapshot.
        self.resume_phase = phase
        self.resume_iteration = int(iteration)

        # Collect RNG state from every rank.
        self.rng_states = (
            self._collect_rng_states()
        )

        # Make sure every rank has reached the checkpoint.
        self.accelerator.wait_for_everyone()

        ckpt_path = None

        if self.accelerator.is_main_process:
            os.makedirs(
                "checkpoints",
                exist_ok=True,
            )

            ckpt = {}

            # Store all expected fields.
            for key in self._keys_to_save:
                value = self.__dict__.get(key)

                if (
                    key.endswith("_optimizer")
                    and value is not None
                ):
                    ckpt[key] = value.state_dict()

                elif hasattr(value, "module"):
                    ckpt[key] = (
                        self.accelerator.unwrap_model(
                            value
                        )
                    )

                else:
                    ckpt[key] = value

            # Extra checkpoint metadata.
            ckpt["checkpoint_version"] = 2

            # Atomic write.
            tmp_path = (
                "checkpoints/model_latest.pth.tmp"
            )

            torch.save(
                ckpt,
                tmp_path,
            )

            os.replace(
                tmp_path,
                "checkpoints/model_latest.pth",
            )

            if phase == "train":
                log.info(
                    "Saved training checkpoint: "
                    f"epoch={self.epoch}, "
                    f"next_batch={iteration}, "
                    f"global_step={self.global_step}"
                )

            elif phase == "train_complete":
                log.info(
                    "Saved train-complete checkpoint: "
                    f"epoch={self.epoch}, "
                    f"global_step={self.global_step}"
                )

            elif phase == "epoch_complete":
                log.info(
                    "Saved epoch-complete checkpoint: "
                    f"epoch={self.epoch}, "
                    f"global_step={self.global_step}"
                )

            # ----------------------------------------------------------
            # Optional numbered epoch checkpoint.
            # ----------------------------------------------------------
            if save_epoch_copy:
                path = (
                    f"checkpoints/model_{self.epoch}.pth"
                )

                torch.save(
                    ckpt,
                    path,
                )

                ckpt_path = os.path.join(
                    os.getcwd(),
                    path,
                )

                log.info(
                    f"Saved epoch checkpoint {path}"
                )

        self.accelerator.wait_for_everyone()

        return (
            ckpt_path,
            self.cfg["saved_folder"].split("/")[-1],
            self.epoch,
        )

    def load_ckpt(
        self,
        filename="model_latest.pth",
    ):
        """
        Load checkpoint and stage optimizer/RNG state for later
        restoration.
        """
        filename = str(filename)

        ckpt = torch.load(
            filename,
            map_location="cpu",
            weights_only=False,
        )

        self._loaded_optim_state = {}

        # --------------------------------------------------------------
        # Validate DataLoader-related configuration before applying
        # checkpoint progress.
        # --------------------------------------------------------------
        self._validate_checkpoint_compatibility(
            ckpt
        )

        for key, value in ckpt.items():
            if (
                key.endswith("_optimizer")
                and isinstance(value, dict)
            ):
                self._loaded_optim_state[key] = value

            elif key == "checkpoint_version":
                # Metadata only.
                continue

            else:
                setattr(
                    self,
                    key,
                    value,
                )

        # --------------------------------------------------------------
        # Old checkpoints
        # --------------------------------------------------------------
        old_checkpoint = (
            ckpt.get("checkpoint_version") is None
        )

        if "resume_phase" not in ckpt:
            self.resume_phase = (
                "epoch_complete"
            )

        if "resume_iteration" not in ckpt:
            self.resume_iteration = 0

        if "target_epoch" not in ckpt:
            # Preserve the old behavior as closely as possible
            # for checkpoints created with the old Trainer.
            self.target_epoch = (
                self.epoch + self.total_epochs
            )

        if "epoch_log" not in ckpt:
            self.epoch_log = OrderedDict()

        # --------------------------------------------------------------
        # RNG state
        # --------------------------------------------------------------
        self._loaded_rng_states = ckpt.get(
            "rng_states"
        )

        # --------------------------------------------------------------
        # Informative warning for old checkpoint format.
        # --------------------------------------------------------------
        if old_checkpoint:
            log.warning(
                "This is an old checkpoint without exact "
                "mid-epoch resume metadata. "
                "It can restore model/optimizer state, "
                "but its exact DataLoader position cannot be recovered."
            )

        missing = (
            set(self._keys_to_save)
            - set(ckpt.keys())
        )

        # These are expected to be absent in old checkpoints.
        expected_old_missing = {
            "resume_phase",
            "resume_iteration",
            "target_epoch",
            "epoch_log",
            "rng_states",
            "train_num_batches",
            "gpu_batch_size",
            "num_processes",
            "encoder_config_seed",
        }

        meaningful_missing = (
            missing
            - expected_old_missing
        )

        if meaningful_missing:
            log.warning(
                f"Keys missing from checkpoint: "
                f"{meaningful_missing}"
            )

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    def log_train_step(
        self,
        loss,
        loss_components,
    ):
        """
        Send one W&B record for every training optimizer step.
        """
        self.global_step += 1

        if not self.accelerator.is_main_process:
            return

        data = {
            "train/loss": float(
                loss.item()
            ),
            "epoch": self.epoch,
        }

        data.update(
            {
                f"train/{key}": float(value)
                for key, value in loss_components.items()
            }
        )

        self.wandb_run.log(
            data,
            step=self.global_step,
        )

    def logs_update(
        self,
        logs,
    ):
        for key, value in logs.items():
            if isinstance(
                value,
                torch.Tensor,
            ):
                value = [
                    value.detach()
                    .cpu()
                    .item()
                ]

            length = len(value)

            count, total = self.epoch_log.get(
                key,
                (0, 0.0),
            )

            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self):
        epoch_log = OrderedDict()

        for key, (count, total) in (
            self.epoch_log.items()
        ):
            if count > 0:
                epoch_log[key] = (
                    total / count
                )

        epoch_log["epoch"] = self.epoch

        if "train_loss" in epoch_log:
            log.info(
                f"Epoch {self.epoch} "
                f"Training loss: "
                f"{epoch_log['train_loss']:.4f} "
                f"Validation loss: "
                f"{epoch_log.get('val_loss', float('nan')):.4f}"
            )

        if self.accelerator.is_main_process:
            self.wandb_run.log(
                {
                    f"epoch/{key}": value
                    for key, value in epoch_log.items()
                },
                step=self.global_step,
            )

        self.epoch_log = OrderedDict()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        full_train_loader = (
            self.dataloaders["train"]
        )

        total_batches = len(
            full_train_loader
        )

        # --------------------------------------------------------------
        # Decide where this epoch starts.
        # --------------------------------------------------------------
        if self.resume_phase == "train":
            start_iteration = int(
                self.resume_iteration
            )

            if start_iteration < 0:
                raise RuntimeError(
                    f"Invalid resume_iteration="
                    f"{start_iteration}"
                )

            if start_iteration > total_batches:
                raise RuntimeError(
                    "Checkpoint points past the end of "
                    "the training DataLoader: "
                    f"{start_iteration} > {total_batches}"
                )

            if start_iteration > 0:
                log.info(
                    f"Resuming epoch {self.epoch} "
                    f"from training batch "
                    f"{start_iteration}/{total_batches} "
                    f"(global_step={self.global_step})"
                )

            # ----------------------------------------------------------
            # Accelerate provides an efficient way to continue from a
            # batch position in a prepared DataLoader.
            #
            # This is preferable to:
            #
            #   for i, data in enumerate(loader):
            #       if i < start_iteration:
            #           continue
            #
            # because the skipped batches never enter the body of the
            # training loop.
            # ----------------------------------------------------------
            train_loader = (
                self.accelerator.skip_first_batches(
                    full_train_loader,
                    num_batches=start_iteration,
                )
            )

        else:
            start_iteration = 0
            train_loader = full_train_loader

            # A new epoch starts with fresh statistics.
            self.resume_iteration = 0

        remaining_batches = max(
            0,
            total_batches - start_iteration,
        )

        progress = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch} Train",
            total=remaining_batches,
        )

        # --------------------------------------------------------------
        # Training loop
        # --------------------------------------------------------------
        for offset, data in enumerate(
            progress
        ):
            # Convert from resumed-loader index back to the ORIGINAL
            # epoch batch index.
            i = (
                start_iteration
                + offset
            )

            obs, act, state = data

            plot = i == 0

            decoder_active = (
                self.decoder_training_active()
            )

            self.model.train_decoder = (
                decoder_active
            )

            self.model.train()

            if self.cfg.has_decoder:
                self.decoder.train(
                    decoder_active
                )

            (
                z_out,
                visual_out,
                visual_reconstructed,
                loss,
                loss_components,
            ) = self.ddp_model(
                obs,
                act,
            )

            self.encoder_optimizer.zero_grad()

            if decoder_active:
                self.decoder_optimizer.zero_grad()

            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()

            if decoder_active:
                self.decoder_optimizer.step()

            if (
                self.cfg.has_predictor
                and self.model.train_predictor
            ):
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = (
                self.accelerator
                .gather_for_metrics(loss)
                .mean()
            )

            loss_components = (
                self.accelerator
                .gather_for_metrics(
                    loss_components
                )
            )

            loss_components = {
                key: value.mean().item()
                for key, value in loss_components.items()
            }

            # ----------------------------------------------------------
            # Epoch statistics
            # ----------------------------------------------------------
            self.logs_update(
                {
                    f"train_{key}": [value]
                    for key, value
                    in loss_components.items()
                }
            )

            self.logs_update(
                {
                    "train_loss": [
                        loss.item()
                    ]
                }
            )

            # ----------------------------------------------------------
            # W&B
            # ----------------------------------------------------------
            self.log_train_step(
                loss,
                loss_components,
            )

            # ----------------------------------------------------------
            # Expensive diagnostics only on first BATCH OF THE EPOCH.
            #
            # This remains false after resuming because i contains the
            # original epoch batch index.
            # ----------------------------------------------------------
            if (
                decoder_active
                and plot
            ):
                if self.cfg.has_predictor:
                    (
                        z_obs_out,
                        z_act_out,
                    ) = self.model.separate_emb(
                        z_out
                    )

                    z_gt = (
                        self.model.encode_obs(
                            obs
                        )
                    )

                    z_tgt = slice_trajdict_with_t(
                        z_gt,
                        start_idx=self.model.num_pred,
                    )

                    err_logs = self.err_eval(
                        z_obs_out,
                        z_tgt,
                    )

                    err_logs = (
                        self.accelerator
                        .gather_for_metrics(
                            err_logs
                        )
                    )

                    err_logs = {
                        key: value.mean().item()
                        for key, value in err_logs.items()
                    }

                    self.logs_update(
                        {
                            f"train_{key}": [
                                value
                            ]
                            for key, value
                            in err_logs.items()
                        }
                    )

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist,
                        (
                            self.cfg.num_hist
                            + self.cfg.num_pred
                        ),
                    ):
                        scores = eval_images(
                            visual_out[
                                :,
                                t
                                - self.cfg.num_pred,
                            ],
                            obs["visual"][
                                :,
                                t,
                            ],
                        )

                        scores = (
                            self.accelerator
                            .gather_for_metrics(
                                scores
                            )
                        )

                        scores = {
                            f"train_img_{key}_pred": [
                                value.mean().item()
                            ]
                            for key, value
                            in scores.items()
                        }

                        self.logs_update(
                            scores
                        )

                if (
                    visual_reconstructed
                    is not None
                ):
                    for t in range(
                        obs["visual"].shape[1]
                    ):
                        scores = eval_images(
                            visual_reconstructed[
                                :,
                                t,
                            ],
                            obs["visual"][
                                :,
                                t,
                            ],
                        )

                        scores = (
                            self.accelerator
                            .gather_for_metrics(
                                scores
                            )
                        )

                        scores = {
                            f"train_img_{key}_reconstructed": [
                                value.mean().item()
                            ]
                            for key, value
                            in scores.items()
                        }

                        self.logs_update(
                            scores
                        )

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    i,
                    self.num_reconstruct_samples,
                    "train",
                )

            # ----------------------------------------------------------
            # EXACT MID-EPOCH CHECKPOINT
            #
            # global_step is incremented inside log_train_step(), so
            # this checkpoint occurs exactly when global_step is a
            # multiple of SAVE_EVERY_ITERS.
            #
            # i+1 = number of batches already completed.
            # Therefore, after restart, batch i+1 is the first one run.
            # ----------------------------------------------------------
            if (
                self.save_every_iters > 0
                and self.global_step
                % self.save_every_iters
                == 0
            ):
                self.save_ckpt(
                    phase="train",
                    iteration=i + 1,
                    save_epoch_copy=False,
                )

        # --------------------------------------------------------------
        # Training epoch has now completed.
        #
        # Save the CURRENT weights and optimizer state before validation.
        #
        # This protects against a crash during validation. On restart,
        # training will NOT be repeated; validation will simply be
        # restarted for this epoch.
        # --------------------------------------------------------------
        self.resume_phase = "train_complete"
        self.resume_iteration = total_batches

        self.save_ckpt(
            phase="train_complete",
            iteration=total_batches,
            save_epoch_copy=False,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def val(self):
        decoder_active = (
            self.decoder_training_active()
        )

        self.model.train_decoder = (
            decoder_active
        )

        self.model.eval()

        if (
            len(self.train_traj_dset) > 0
            and self.cfg.has_predictor
        ):
            train_rollout_logs = (
                self.openloop_rollout(
                    self.train_traj_dset,
                    mode="train",
                )
            )

            self.logs_update(
                {
                    f"train_{key}": [value]
                    for key, value
                    in train_rollout_logs.items()
                }
            )

            val_rollout_logs = (
                self.openloop_rollout(
                    self.val_traj_dset,
                    mode="val",
                )
            )

            self.logs_update(
                {
                    f"val_{key}": [value]
                    for key, value
                    in val_rollout_logs.items()
                }
            )

        self.accelerator.wait_for_everyone()

        for i, data in enumerate(
            tqdm(
                self.dataloaders["valid"],
                desc=f"Epoch {self.epoch} Valid",
            )
        ):
            obs, act, state = data

            plot = i == 0

            self.model.eval()

            (
                z_out,
                visual_out,
                visual_reconstructed,
                loss,
                loss_components,
            ) = self.model(
                obs,
                act,
            )

            loss = (
                self.accelerator
                .gather_for_metrics(loss)
                .mean()
            )

            loss_components = (
                self.accelerator
                .gather_for_metrics(
                    loss_components
                )
            )

            loss_components = {
                key: value.mean().item()
                for key, value in loss_components.items()
            }

            self.logs_update(
                {
                    "val_loss": [
                        loss.item()
                    ]
                }
            )

            self.logs_update(
                {
                    f"val_{key}": [value]
                    for key, value
                    in loss_components.items()
                }
            )

            if (
                decoder_active
                and plot
            ):
                if self.cfg.has_predictor:
                    (
                        z_obs_out,
                        z_act_out,
                    ) = self.model.separate_emb(
                        z_out
                    )

                    z_gt = (
                        self.model.encode_obs(
                            obs
                        )
                    )

                    z_tgt = slice_trajdict_with_t(
                        z_gt,
                        start_idx=self.model.num_pred,
                    )

                    err_logs = self.err_eval(
                        z_obs_out,
                        z_tgt,
                    )

                    err_logs = (
                        self.accelerator
                        .gather_for_metrics(
                            err_logs
                        )
                    )

                    err_logs = {
                        key: value.mean().item()
                        for key, value
                        in err_logs.items()
                    }

                    self.logs_update(
                        {
                            f"val_{key}": [
                                value
                            ]
                            for key, value
                            in err_logs.items()
                        }
                    )

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist,
                        (
                            self.cfg.num_hist
                            + self.cfg.num_pred
                        ),
                    ):
                        scores = eval_images(
                            visual_out[
                                :,
                                t
                                - self.cfg.num_pred,
                            ],
                            obs["visual"][
                                :,
                                t,
                            ],
                        )

                        scores = (
                            self.accelerator
                            .gather_for_metrics(
                                scores
                            )
                        )

                        self.logs_update(
                            {
                                f"val_img_{key}_pred": [
                                    value.mean().item()
                                ]
                                for key, value
                                in scores.items()
                            }
                        )

                if (
                    visual_reconstructed
                    is not None
                ):
                    for t in range(
                        obs["visual"].shape[1]
                    ):
                        scores = eval_images(
                            visual_reconstructed[
                                :,
                                t,
                            ],
                            obs["visual"][
                                :,
                                t,
                            ],
                        )

                        scores = (
                            self.accelerator
                            .gather_for_metrics(
                                scores
                            )
                        )

                        self.logs_update(
                            {
                                f"val_img_{key}_reconstructed": [
                                    value.mean().item()
                                ]
                                for key, value
                                in scores.items()
                            }
                        )

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    i,
                    self.num_reconstruct_samples,
                    "valid",
                )

    # ------------------------------------------------------------------
    # Rollouts
    # ------------------------------------------------------------------

    def openloop_rollout(
        self,
        dset,
        num_rollout=10,
        rand_start_end=True,
        min_horizon=2,
        mode="train",
    ):
        np.random.seed(
            self.cfg.training.seed
        )

        min_horizon += (
            self.cfg.num_hist
        )

        plotting_dir = (
            f"rollout_plots/"
            f"e{self.epoch}_rollout"
        )

        if self.accelerator.is_main_process:
            os.makedirs(
                plotting_dir,
                exist_ok=True,
            )

        self.accelerator.wait_for_everyone()

        logs = {}

        num_past = [
            (
                self.cfg.num_hist,
                "",
            ),
            (
                1,
                "_1framestart",
            ),
        ]

        for idx in range(
            num_rollout
        ):
            valid_traj = False

            while not valid_traj:
                traj_idx = np.random.randint(
                    0,
                    len(dset),
                )

                obs, act, state, _ = dset[
                    traj_idx
                ]

                act = act.to(
                    self.device
                )

                if rand_start_end:
                    if (
                        obs["visual"].shape[0]
                        > min_horizon
                        * self.cfg.frameskip
                        + 1
                    ):
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0]
                            - min_horizon
                            * self.cfg.frameskip
                            - 1,
                        )

                    else:
                        start = 0

                    max_horizon = (
                        obs["visual"].shape[0]
                        - start
                        - 1
                    ) // self.cfg.frameskip

                    if (
                        max_horizon
                        > min_horizon
                    ):
                        valid_traj = True

                        horizon = np.random.randint(
                            min_horizon,
                            max_horizon + 1,
                        )

                else:
                    valid_traj = True
                    start = 0

                    horizon = (
                        obs["visual"].shape[0]
                        - 1
                    ) // self.cfg.frameskip

            for key in obs:
                obs[key] = obs[key][
                    start:
                    start
                    + horizon
                    * self.cfg.frameskip
                    + 1:
                    self.cfg.frameskip
                ]

            act = act[
                start:
                start
                + horizon
                * self.cfg.frameskip
            ]

            act = rearrange(
                act,
                "(h f) d -> h (f d)",
                f=self.cfg.frameskip,
            )

            obs_g = {
                key: obs[key][-1]
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
                for key in obs
            }

            z_g = self.model.encode_obs(
                obs_g
            )

            actions = act.unsqueeze(0)

            for n_past, postfix in num_past:
                obs_0 = {
                    key: obs[key][
                        :n_past
                    ]
                    .unsqueeze(0)
                    .to(self.device)
                    for key in obs
                }

                z_obses, z = (
                    self.model.rollout(
                        obs_0,
                        actions,
                    )
                )

                z_obs_last = (
                    slice_trajdict_with_t(
                        z_obses,
                        start_idx=-1,
                        end_idx=None,
                    )
                )

                div_loss = (
                    self.err_eval_single(
                        z_obs_last,
                        z_g,
                    )
                )

                for key, value in (
                    div_loss.items()
                ):
                    log_key = (
                        f"z_{key}_err_rollout"
                        f"{postfix}"
                    )

                    logs.setdefault(
                        log_key,
                        [],
                    ).append(value)

                if self.cfg.has_decoder:
                    visuals = (
                        self.model
                        .decode_obs(z_obses)[0][
                            "visual"
                        ]
                    )

                    imgs = torch.cat(
                        [
                            obs["visual"],
                            visuals[0].cpu(),
                        ],
                        dim=0,
                    )

                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        (
                            f"{plotting_dir}/"
                            f"e{self.epoch}_"
                            f"{mode}_{idx}"
                            f"{postfix}.png"
                        ),
                    )

        return {
            key: sum(values)
            / len(values)
            for key, values
            in logs.items()
            if values
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        num_frames = (
            gt_imgs.shape[1]
        )

        (
            gt_imgs,
            pred_imgs,
            reconstructed_gt_imgs,
        ) = sample_tensors(
            [
                gt_imgs,
                pred_imgs,
                reconstructed_gt_imgs,
            ],
            num_samples,
            indices=list(
                range(num_samples)
            )[:gt_imgs.shape[0]],
        )

        num_samples = min(
            num_samples,
            gt_imgs.shape[0],
        )

        if pred_imgs is not None:
            pred_imgs = torch.cat(
                [
                    torch.full(
                        (
                            num_samples,
                            self.model.num_pred,
                            *pred_imgs.shape[2:],
                        ),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ],
                dim=1,
            )

        else:
            pred_imgs = torch.full(
                gt_imgs.shape,
                -1,
                device=self.device,
            )

        pred_imgs = rearrange(
            pred_imgs,
            "b t c h w -> (b t) c h w",
        )

        gt_imgs = rearrange(
            gt_imgs,
            "b t c h w -> (b t) c h w",
        )

        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs,
            "b t c h w -> (b t) c h w",
        )

        imgs = torch.cat(
            [
                gt_imgs,
                pred_imgs,
                reconstructed_gt_imgs,
            ],
            dim=0,
        )

        if self.accelerator.is_main_process:
            os.makedirs(
                phase,
                exist_ok=True,
            )

        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_samples * num_frames,
            (
                f"{phase}/"
                f"{phase}_e{epoch:05d}_"
                f"b{batch}.png"
            ),
        )

    def plot_imgs(
        self,
        imgs,
        num_columns,
        img_name,
    ):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )

    # ------------------------------------------------------------------
    # Planning jobs
    # ------------------------------------------------------------------

    def monitor_jobs(
        self,
        lock,
    ):
        while True:
            with lock:
                finished_jobs = [
                    job_tuple
                    for job_tuple in self.job_set
                    if job_tuple[2].done()
                ]

                for (
                    epoch,
                    job_name,
                    job,
                ) in finished_jobs:
                    result = job.result()

                    log_data = {
                        f"{job_name}/{key}": value
                        for key, value
                        in result.items()
                    }

                    log_data["epoch"] = epoch

                    self.wandb_run.log(
                        log_data,
                        step=self.global_step,
                    )

                    self.job_set.remove(
                        (
                            epoch,
                            job_name,
                            job,
                        )
                    )

            time.sleep(1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        # --------------------------------------------------------------
        # Planning workers
        # --------------------------------------------------------------
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(
                max_workers=4
            )

            self.job_set = set()

            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs,
                args=(lock,),
                daemon=True,
            )

            self.monitor_thread.start()

        # --------------------------------------------------------------
        # Determine where to resume.
        # --------------------------------------------------------------
        if self.resume_phase in {
            "train",
            "train_complete",
        }:
            init_epoch = self.epoch
        else:
            init_epoch = self.epoch + 1

        # --------------------------------------------------------------
        # target_epoch is the ORIGINAL final epoch.
        #
        # Fresh run:
        #   epoch 0 + cfg.training.epochs
        #
        # Resume from epoch 20 of a 100 epoch run:
        #   continue until epoch 100, NOT 120.
        # --------------------------------------------------------------
        end_epoch = int(
            self.target_epoch
        )

        if init_epoch > end_epoch:
            log.info(
                f"Training already complete: "
                f"init_epoch={init_epoch}, "
                f"target_epoch={end_epoch}"
            )

            if self.accelerator.is_main_process:
                try:
                    executor.shutdown(
                        wait=False
                    )
                except Exception:
                    pass

            return

        if self.accelerator.is_main_process:
            log.info(
                f"Starting training from "
                f"epoch={init_epoch} "
                f"through epoch={end_epoch}"
            )

            if self.resume_phase == "train":
                log.info(
                    f"Exact mid-epoch resume: "
                    f"epoch={self.epoch}, "
                    f"next_train_batch="
                    f"{self.resume_iteration}, "
                    f"global_step={self.global_step}"
                )

            elif (
                self.resume_phase
                == "train_complete"
            ):
                log.info(
                    f"Training for epoch {self.epoch} "
                    f"was already complete; "
                    f"resuming at validation."
                )

        # --------------------------------------------------------------
        # Epoch loop
        # --------------------------------------------------------------
        for epoch in range(
            init_epoch,
            end_epoch + 1,
        ):
            self.epoch = epoch

            # ----------------------------------------------------------
            # Determine whether this is a true resume of the current
            # epoch or a completely new epoch.
            # ----------------------------------------------------------
            continuing_current_epoch = (
                epoch == init_epoch
                and self.resume_phase
                in {
                    "train",
                    "train_complete",
                }
            )

            if not continuing_current_epoch:
                self.resume_phase = "train"
                self.resume_iteration = 0
                self.epoch_log = OrderedDict()

            if self.accelerator.is_main_process:
                log.info(
                    f"Epoch {self.epoch}: "
                    f"decoder_active="
                    f"{self.decoder_training_active()}"
                )

            self.accelerator.wait_for_everyone()

            # ----------------------------------------------------------
            # TRAIN
            # ----------------------------------------------------------
            if self.resume_phase == "train":
                self.train()

            elif (
                self.resume_phase
                == "train_complete"
            ):
                log.info(
                    f"Skipping training for epoch "
                    f"{self.epoch}; "
                    f"checkpoint says training is "
                    f"already complete."
                )

            else:
                # Defensive fallback.
                self.train()

            self.accelerator.wait_for_everyone()

            # ----------------------------------------------------------
            # We are now in the validation phase.
            #
            # If training had just completed, this checkpoint has
            # already been written by train().
            # ----------------------------------------------------------
            self.resume_phase = "train_complete"
            self.resume_iteration = (
                self.train_num_batches
            )

            self.accelerator.wait_for_everyone()

            # ----------------------------------------------------------
            # VALIDATION
            # ----------------------------------------------------------
            self.val()

            self.accelerator.wait_for_everyone()

            # ----------------------------------------------------------
            # Epoch-level logging
            # ----------------------------------------------------------
            self.logs_flash()

            self.accelerator.wait_for_everyone()

            # ----------------------------------------------------------
            # Mark the whole epoch complete.
            #
            # IMPORTANT:
            # We update model_latest after EVERY completed epoch.
            #
            # This means that even if save_every_x_epoch=10, an
            # interruption after validation does NOT fall back to a
            # checkpoint from 10 epochs ago.
            #
            # model_{epoch}.pth is still only created according to the
            # user's configured save_every_x_epoch.
            # ----------------------------------------------------------
            save_epoch_copy = (
                self.epoch
                % self.cfg.training.save_every_x_epoch
                == 0
            )

            self.save_ckpt(
                phase="epoch_complete",
                iteration=0,
                save_epoch_copy=save_epoch_copy,
            )

            # ----------------------------------------------------------
            # Planning jobs
            # ----------------------------------------------------------
            if (
                self.cfg.plan_settings.plan_cfg_path
                is not None
            ):
                # We only launch planning jobs when a numbered epoch
                # checkpoint exists, matching the previous behavior.
                if (
                    self.accelerator.is_main_process
                    and save_epoch_copy
                ):
                    from plan import (
                        build_plan_cfg_dicts,
                        launch_plan_jobs,
                    )

                    ckpt_path = os.path.join(
                        os.getcwd(),
                        "checkpoints",
                        f"model_{self.epoch}.pth",
                    )

                    model_name = (
                        self.cfg[
                            "saved_folder"
                        ]
                        .split("/")[-1]
                    )

                    model_epoch = self.epoch

                    cfg_dicts = (
                        build_plan_cfg_dicts(
                            plan_cfg_path=os.path.join(
                                self.base_path,
                                self.cfg.plan_settings.plan_cfg_path,
                            ),
                            ckpt_base_path=(
                                self.cfg.ckpt_base_path
                            ),
                            model_name=model_name,
                            model_epoch=model_epoch,
                            planner=(
                                self.cfg.plan_settings.planner
                            ),
                            goal_source=(
                                self.cfg.plan_settings.goal_source
                            ),
                            goal_H=(
                                self.cfg.plan_settings.goal_H
                            ),
                            alpha=(
                                self.cfg.plan_settings.alpha
                            ),
                        )
                    )

                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(),
                            "submitit-evals",
                            f"epoch_{self.epoch}",
                        ),
                    )

                    with lock:
                        self.job_set.update(
                            jobs
                        )

            # ----------------------------------------------------------
            # After a complete epoch, the next epoch must begin from
            # batch 0.
            # ----------------------------------------------------------
            self.resume_phase = (
                "epoch_complete"
            )

            self.resume_iteration = 0
            self.epoch_log = OrderedDict()

            self.accelerator.wait_for_everyone()

        # --------------------------------------------------------------
        # Final cleanup
        # --------------------------------------------------------------
        if self.accelerator.is_main_process:
            log.info(
                f"Training finished at "
                f"epoch={self.epoch}, "
                f"global_step={self.global_step}"
            )

            try:
                executor.shutdown(
                    wait=False
                )
            except Exception:
                pass


@hydra.main(
    config_path="conf",
    config_name="train",
)
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
