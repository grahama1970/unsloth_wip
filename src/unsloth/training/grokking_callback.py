"""Callback for monitoring and optimizing grokking during training."""
Module: grokking_callback.py

import torch
from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState

from ..core.grokking_config import GrokkingConfig, GrokkingMonitor


class GrokkingCallback(TrainerCallback):
    """Monitor and optimize training for grokking phenomenon."""

    def __init__(
        self,
        grokking_config: GrokkingConfig,
        writer=None,  # TensorBoard writer
        base_weight_decay: float = 0.01
    ):
        self.config = grokking_config
        self.writer = writer
        self.monitor = GrokkingMonitor()
        self.base_weight_decay = base_weight_decay
        self.phase_logged = set()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Monitor training for grokking phases."""
        if not logs or not self.config.track_memorization:
            return

        # Extract losses
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)

        if train_loss is not None and eval_loss is not None:
            # Update monitor
            phase_change = self.monitor.update(
                train_loss, eval_loss, state.global_step, self.config
            )

            # Log phase transitions
            if phase_change != "no_change" and phase_change not in self.phase_logged:
                logger.info(f" Grokking phase transition: {phase_change} at step {state.global_step}")
                self.phase_logged.add(phase_change)

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_text(
                        "grokking/phase_transition",
                        f"{phase_change} at step {state.global_step}",
                        state.global_step
                    )

                # Save checkpoint at memorization phase
                if (phase_change == "entered_memorization" and
                    self.config.save_memorization_checkpoint):
                    control.should_save = True
                    logger.info(" Saving memorization checkpoint")

            # Log grokking metrics
            if self.writer:
                summary = self.monitor.get_summary()
                self.writer.add_scalar(
                    "grokking/phase",
                    {"initialization": 0, "memorization": 1, "grokking": 2, "converged": 3}.get(
                        summary["current_phase"], -1
                    ),
                    state.global_step
                )

                # Log train/val gap
                if train_loss and eval_loss:
                    self.writer.add_scalar(
                        "grokking/train_val_gap",
                        eval_loss - train_loss,
                        state.global_step
                    )

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Apply grokking-specific optimizations during training."""
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")

        if not model or not optimizer:
            return

        # Update weight decay based on schedule
        if self.config.grokking_weight_decay_schedule != "constant":
            total_steps = args.max_steps or (args.num_train_epochs * state.steps_per_epoch)
            new_weight_decay = self.config.get_weight_decay_schedule(
                state.global_step, total_steps
            )

            # Update optimizer weight decay
            for group in optimizer.param_groups:
                group['weight_decay'] = new_weight_decay

            if self.writer and state.global_step % 100 == 0:
                self.writer.add_scalar(
                    "grokking/weight_decay",
                    new_weight_decay,
                    state.global_step
                )

        # Apply gradient noise if configured
        if self.config.grokking_gradient_noise > 0:
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param.grad) * self.config.grokking_gradient_noise
                        param.grad.add_(noise)

        # Layer swapping (experimental technique from recent research)
        if (self.config.use_layer_swapping and
            state.global_step % self.config.layer_swap_interval == 0 and
            state.global_step > 0):
            self._swap_random_layers(model)
            logger.info(f" Performed layer swapping at step {state.global_step}")

    def _swap_random_layers(self, model):
        """Swap weights between random LoRA layers to encourage exploration."""
        import random

        # Get all LoRA modules
        lora_modules = []
        for name, module in model.named_modules():
            if "lora" in name.lower() and hasattr(module, "weight"):
                lora_modules.append((name, module))

        if len(lora_modules) >= 2:
            # Randomly select two modules to swap
            idx1, idx2 = random.sample(range(len(lora_modules)), 2)
            module1_name, module1 = lora_modules[idx1]
            module2_name, module2 = lora_modules[idx2]

            # Swap weights
            with torch.no_grad():
                if hasattr(module1, "weight") and hasattr(module2, "weight"):
                    if module1.weight.shape == module2.weight.shape:
                        temp = module1.weight.data.clone()
                        module1.weight.data = module2.weight.data.clone()
                        module2.weight.data = temp

                        if self.writer:
                            self.writer.add_text(
                                "grokking/layer_swap",
                                f"Swapped {module1_name} <-> {module2_name}",
                                self.monitor.grokking_start_step or 0
                            )

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log final grokking summary."""
        summary = self.monitor.get_summary()

        logger.info(" Grokking Training Summary:")
        logger.info(f"   Final phase: {summary['current_phase']}")

        if summary['memorization_start_step']:
            logger.info(f"   Memorization started: step {summary['memorization_start_step']}")

        if summary['grokking_start_step']:
            logger.info(f"   Grokking started: step {summary['grokking_start_step']}")

        if summary['steps_in_memorization']:
            logger.info(f"   Steps in memorization: {summary['steps_in_memorization']}")

        # Log to TensorBoard
        if self.writer:
            self.writer.add_text(
                "grokking/final_summary",
                str(summary),
                state.global_step
            )
