"""Configuration for grokking-based training strategies.

Based on research showing that extended training with proper regularization
can lead to delayed but superior generalization (grokking phenomenon).

References:
- Grokking: Generalization Beyond Overfitting (Power et al., 2022)
- Recent advances in grokking mechanisms (2025)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import math


@dataclass
class GrokkingConfig:
    """Configuration for grokking-based training.
    
    Grokking involves extended training (30×+ normal epochs) with strong
    regularization, leading to initial memorization followed by sudden
    generalization after many epochs.
    """
    
    # Grokking enablement
    enable_grokking: bool = False
    grokking_multiplier: float = 30.0  # Multiply normal epochs by this
    
    # Weight decay is crucial for grokking
    grokking_weight_decay: float = 0.1  # Higher than normal (0.01)
    grokking_weight_decay_schedule: str = "constant"  # "constant", "cosine", "linear"
    
    # Learning rate scheduling for grokking
    grokking_lr_schedule: str = "cosine_with_restarts"  # Better for long training
    grokking_lr_min: float = 1e-6  # Lower final LR for grokking
    grokking_warmup_ratio: float = 0.02  # Shorter warmup for extended training
    
    # Regularization strategies
    grokking_dropout_schedule: str = "constant"  # "constant", "linear_decrease", "cosine"
    grokking_label_smoothing: float = 0.1  # Label smoothing helps grokking
    grokking_gradient_noise: float = 0.01  # Add gradient noise for exploration
    
    # Monitoring grokking phases
    track_memorization: bool = True  # Track train/val divergence
    memorization_threshold: float = 0.5  # Loss difference indicating memorization
    grokking_patience: int = 1000  # Steps to wait after memorization
    
    # Advanced grokking techniques
    use_layer_swapping: bool = False  # Swap layers during training
    layer_swap_interval: int = 5000  # Steps between layer swaps
    use_grokfast: bool = False  # Use Grokfast acceleration method
    
    # Early stopping override for grokking
    disable_early_stopping: bool = True  # Grokking requires patience
    save_memorization_checkpoint: bool = True  # Save model at memorization phase
    
    # Validation monitoring
    track_validation_phases: bool = True
    validation_window: int = 100  # Steps to average for phase detection
    phase_change_threshold: float = 0.05  # Relative change to detect phase
    
    def calculate_grokking_epochs(self, base_epochs: int) -> int:
        """Calculate total epochs for grokking training."""
        return int(base_epochs * self.grokking_multiplier)
        
    def get_weight_decay_schedule(self, current_step: int, total_steps: int) -> float:
        """Get weight decay value based on schedule."""
        if self.grokking_weight_decay_schedule == "constant":
            return self.grokking_weight_decay
        elif self.grokking_weight_decay_schedule == "cosine":
            return self.grokking_weight_decay * (
                0.5 * (1 + math.cos(math.pi * current_step / total_steps))
            )
        elif self.grokking_weight_decay_schedule == "linear":
            return self.grokking_weight_decay * (1 - current_step / total_steps)
        else:
            return self.grokking_weight_decay
            
    def get_dropout_rate(self, base_dropout: float, current_step: int, total_steps: int) -> float:
        """Get dropout rate based on schedule."""
        if self.grokking_dropout_schedule == "constant":
            return base_dropout
        elif self.grokking_dropout_schedule == "linear_decrease":
            # Decrease dropout over time to allow more complex patterns
            return base_dropout * (1 - 0.5 * current_step / total_steps)
        elif self.grokking_dropout_schedule == "cosine":
            return base_dropout * (0.5 * (1 + math.cos(math.pi * current_step / total_steps)))
        else:
            return base_dropout


@dataclass
class GrokkingMonitor:
    """Monitor training phases for grokking detection."""
    
    phase: str = "initialization"  # "initialization", "memorization", "grokking", "converged"
    memorization_start_step: Optional[int] = None
    grokking_start_step: Optional[int] = None
    
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    
    def update(self, train_loss: float, val_loss: float, step: int, config: GrokkingConfig) -> str:
        """Update monitoring and detect phase transitions."""
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        
        # Keep only recent history
        window = config.validation_window
        if len(self.train_loss_history) > window:
            self.train_loss_history = self.train_loss_history[-window:]
            self.val_loss_history = self.val_loss_history[-window:]
            
        # Detect phase transitions
        if len(self.train_loss_history) >= 10:
            avg_train = sum(self.train_loss_history[-10:]) / 10
            avg_val = sum(self.val_loss_history[-10:]) / 10
            
            # Check for memorization (train << val)
            if self.phase == "initialization" and (avg_val - avg_train) > config.memorization_threshold:
                self.phase = "memorization"
                self.memorization_start_step = step
                return "entered_memorization"
                
            # Check for grokking (val starts decreasing after memorization)
            if self.phase == "memorization" and len(self.val_loss_history) >= 20:
                recent_val = sum(self.val_loss_history[-10:]) / 10
                older_val = sum(self.val_loss_history[-20:-10]) / 10
                
                if recent_val < older_val * (1 - config.phase_change_threshold):
                    self.phase = "grokking"
                    self.grokking_start_step = step
                    return "entered_grokking"
                    
            # Check for convergence
            if self.phase == "grokking" and len(self.val_loss_history) >= 20:
                val_std = self._calculate_std(self.val_loss_history[-20:])
                if val_std < 0.001:  # Very stable validation loss
                    self.phase = "converged"
                    return "converged"
                    
        return "no_change"
        
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of grokking phases."""
        return {
            "current_phase": self.phase,
            "memorization_start_step": self.memorization_start_step,
            "grokking_start_step": self.grokking_start_step,
            "steps_in_memorization": (
                self.grokking_start_step - self.memorization_start_step
                if self.memorization_start_step and self.grokking_start_step
                else None
            ),
            "latest_train_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "latest_val_loss": self.val_loss_history[-1] if self.val_loss_history else None,
        }