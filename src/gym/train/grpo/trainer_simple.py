"""
Simplified GRPO Trainer - Group Relative Policy Optimization
Copyright 2025 Zoo Labs Foundation Inc.

Minimal implementation following Go/Plan 9 principles:
- No value network (40-60% memory savings)
- Group-based advantage estimation
- Explicit, simple operations
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Optional, Union, Dict, Any

class GRPOTrainer(Trainer):
    """Simplified Group Relative Policy Optimization"""

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
        # Store essentials
        self.ref_model = ref_model
        self.tokenizer = tokenizer

        # Fixed hyperparameters
        self.group_size = 8
        self.beta = 0.1
        self.clip_range = 0.2

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

        # Disable cache
        if hasattr(model, "config"):
            model.config.use_cache = False

        # Prepare reference model
        if ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
            self.ref_model.eval()

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute relative advantages within groups.
        Simple implementation: subtract group mean.
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)

        # Process each group
        for i in range(0, batch_size, self.group_size):
            end = min(i + self.group_size, batch_size)
            group = rewards[i:end]

            # Relative advantage = reward - group mean
            group_mean = group.mean()
            advantages[i:end] = group - group_mean

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        return advantages

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute GRPO loss - no value network needed.

        Invariants:
        - Group-relative advantages
        - Clipped policy gradient
        - No value function
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels", input_ids)

        # Get policy outputs
        policy_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # Use cross-entropy loss as reward signal (negative for maximization)
        # In production, use actual reward model
        rewards = -policy_outputs.loss.detach()

        # Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards)

        # Get log probabilities from policy
        policy_logprobs = -policy_outputs.loss

        # Get reference log probabilities
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                ref_logprobs = -ref_outputs.loss
        else:
            ref_logprobs = policy_logprobs.detach()

        # Compute probability ratio
        log_ratio = policy_logprobs - ref_logprobs
        ratio = torch.exp(log_ratio)

        # Clip ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

        # Policy gradient loss (PPO-style clipping)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        loss = torch.max(pg_loss1, pg_loss2).mean()

        if return_outputs:
            metrics = {
                "loss": loss.item(),
                "advantages_mean": advantages.mean().item(),
                "ratio_mean": ratio.mean().item()
            }
            return loss, metrics

        return loss