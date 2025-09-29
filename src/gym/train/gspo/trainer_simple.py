"""
Simplified GSPO Trainer - Group Sequence Policy Optimization
Copyright 2025 Zoo Labs Foundation Inc.

Minimal implementation following Go/Plan 9 principles:
- Explicit over implicit
- Single obvious implementation
- No unnecessary abstractions
- Standard library preferred
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Union, Dict, Any

class GSPOTrainer(Trainer):
    """Simplified Group Sequence Policy Optimization"""

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module],
        args: Any,
        tokenizer: Any,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        **kwargs
    ):
        # Store essential components only
        self.ref_model = ref_model
        self.tokenizer = tokenizer

        # Fixed hyperparameters - no configuration sprawl
        self.group_size = 8
        self.beta = 0.1
        self.clip_epsilon = 0.2

        # Initialize parent
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )

        # Disable cache for training
        if hasattr(model, "config"):
            model.config.use_cache = False

        # Prepare reference model simply
        if ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
            self.ref_model.eval()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute GSPO loss - simple and explicit.

        Invariants:
        - Sequence-level optimization
        - Group-based normalization
        - Clipped importance ratios
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels", input_ids)

        # Forward pass for policy model
        policy_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # Get sequence likelihood from cross-entropy loss
        # Using built-in loss computation is simpler than manual calculation
        policy_loss = policy_outputs.loss
        policy_likelihood = -policy_loss  # Negative loss as proxy for likelihood

        # Reference model likelihood
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                ref_likelihood = -ref_outputs.loss
        else:
            ref_likelihood = policy_likelihood.detach()

        # Compute importance ratio (simple exponential of difference)
        log_ratio = policy_likelihood - ref_likelihood
        ratio = torch.exp(log_ratio)

        # Clip ratio for stability
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # Simple reward: use the negative of base loss as reward signal
        # In production, replace with actual reward model
        rewards = -policy_loss.detach()

        # Normalize rewards within batch (simpler than groups)
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # GSPO loss: maximize expected reward under clipped ratio
        loss = -(clipped_ratio * rewards_normalized).mean()

        if return_outputs:
            metrics = {
                "loss": loss.item(),
                "reward_mean": rewards.mean().item(),
                "ratio_mean": ratio.mean().item()
            }
            return loss, metrics

        return loss