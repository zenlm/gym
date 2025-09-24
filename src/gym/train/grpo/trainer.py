# Copyright 2025 Zoo Labs Foundation Inc. and the Gym team.
#
# Group Relative Policy Optimization (GRPO) Trainer
# Based on DeepSeek's GRPO: https://arxiv.org/abs/2502.01155
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing_extensions import override
from tqdm import tqdm

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        ProcessorMixin,
        TrainingArguments,
        TrainerCallback,
    )
    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments

logger = logging.get_logger(__name__)


class GRPOTrainer(Trainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer
    
    GRPO eliminates the value network used in PPO, reducing memory consumption by 40-60%.
    It uses group sampling and relative advantage estimation within groups.
    """
    
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        args: "TrainingArguments",
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
        data_collator: Optional[Any] = None,
        callbacks: Optional[list["TrainerCallback"]] = None,
        **kwargs,
    ) -> None:
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        self.ref_model = ref_model
        
        # GRPO specific parameters
        self.group_size = getattr(finetuning_args, "grpo_group_size", 8)
        self.beta = getattr(finetuning_args, "grpo_beta", 0.1)
        self.clip_range = getattr(finetuning_args, "grpo_clip_range", 0.2)
        self.normalize_advantages = getattr(finetuning_args, "grpo_normalize_advantages", True)
        
        # Initialize storage for metrics
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.current_device = get_current_device()
        
        # Call parent constructor
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            **kwargs,
        )
        
        # Disable dropout for stable training
        if hasattr(model, "config"):
            model.config.use_cache = False
            if hasattr(model.config, "dropout"):
                model.config.dropout = 0.0
        
        # Prepare reference model if provided
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) 
                    or getattr(ref_model, "is_loaded_in_4bit", False)
                ):
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()
        
        # Add processor callback if needed
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))
        
        # Add BAdam support if enabled
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version
            self.accelerator.clip_grad_norm_ = lambda *args, **kwargs: clip_grad_norm_old_version(
                self.accelerator, *args, **kwargs
            )
            self.add_callback(BAdamCallback)
        
        warnings.simplefilter("ignore")  # Remove GC warnings
    
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()
    
    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute relative advantages within groups.
        
        Args:
            rewards: Tensor of rewards [batch_size]
            group_size: Size of each group for relative advantage computation
            
        Returns:
            advantages: Tensor of relative advantages [batch_size]
        """
        if group_size is None:
            group_size = self.group_size
        
        batch_size = rewards.shape[0]
        num_groups = math.ceil(batch_size / group_size)
        
        advantages = torch.zeros_like(rewards)
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, batch_size)
            
            group_rewards = rewards[start_idx:end_idx]
            
            # Compute relative advantages within the group
            group_mean = group_rewards.mean()
            group_advantages = group_rewards - group_mean
            
            if self.normalize_advantages and group_advantages.std() > 1e-8:
                group_advantages = group_advantages / (group_advantages.std() + 1e-8)
            
            advantages[start_idx:end_idx] = group_advantages
        
        return advantages
    
    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute GRPO policy loss.
        
        Args:
            logprobs: Log probabilities from policy model
            ref_logprobs: Log probabilities from reference model
            advantages: Computed advantages
            mask: Optional mask for valid tokens
            
        Returns:
            loss: GRPO policy loss
        """
        # Compute log ratios
        log_ratios = logprobs - ref_logprobs
        ratios = torch.exp(log_ratios)
        
        # Clipped objective
        clipped_ratios = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range)
        
        # Policy gradient loss
        pg_loss = -torch.min(
            ratios * advantages,
            clipped_ratios * advantages
        )
        
        if mask is not None:
            pg_loss = pg_loss * mask
            pg_loss = pg_loss.sum() / mask.sum()
        else:
            pg_loss = pg_loss.mean()
        
        return pg_loss
    
    def generate_responses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate responses using the policy model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            responses: Generated response token IDs
            response_mask: Mask for generated tokens
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate responses
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.generating_args.max_new_tokens,
                temperature=self.generating_args.temperature,
                top_p=self.generating_args.top_p,
                do_sample=self.generating_args.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated responses
        prompt_length = input_ids.shape[1]
        responses = outputs[:, prompt_length:]
        
        # Create mask for generated tokens
        response_mask = (responses != self.tokenizer.pad_token_id).float()
        
        self.model.train()
        
        return responses, response_mask
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        responses: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards for generated responses.
        This is a placeholder - should be replaced with actual reward computation.
        
        Args:
            input_ids: Input token IDs
            responses: Generated response token IDs
            
        Returns:
            rewards: Computed rewards
        """
        # Placeholder reward computation
        # In practice, this would call a reward model or use human feedback
        batch_size = input_ids.shape[0]
        
        # For demo purposes, return random rewards
        rewards = torch.randn(batch_size, device=input_ids.device)
        
        return rewards
    
    @override
    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Compute GRPO loss for a batch of inputs.
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate responses
        responses, response_mask = self.generate_responses(input_ids, attention_mask)
        
        # Compute rewards
        rewards = self.compute_rewards(input_ids, responses)
        
        # Compute group advantages
        advantages = self.compute_group_advantages(rewards)
        
        # Concatenate inputs and responses
        full_input_ids = torch.cat([input_ids, responses], dim=1)
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)
        
        # Get logprobs from policy model
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            policy_outputs = model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            policy_logits = policy_outputs.logits
        
        # Compute log probabilities
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        
        # Get reference log probabilities if ref model exists
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        else:
            # If no reference model, use a detached copy of policy logprobs
            ref_logprobs = policy_logprobs.detach()
        
        # Select log probabilities for generated tokens
        prompt_length = input_ids.shape[1]
        response_logprobs = policy_logprobs[:, prompt_length-1:-1, :]
        ref_response_logprobs = ref_logprobs[:, prompt_length-1:-1, :]
        
        # Gather logprobs for actual response tokens
        response_token_logprobs = torch.gather(
            response_logprobs,
            dim=2,
            index=responses.unsqueeze(-1)
        ).squeeze(-1)
        
        ref_response_token_logprobs = torch.gather(
            ref_response_logprobs,
            dim=2,
            index=responses.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum over sequence dimension
        response_token_logprobs = (response_token_logprobs * response_mask).sum(dim=1)
        ref_response_token_logprobs = (ref_response_token_logprobs * response_mask).sum(dim=1)
        
        # Compute policy loss
        loss = self.compute_policy_loss(
            response_token_logprobs,
            ref_response_token_logprobs,
            advantages
        )
        
        # Store metrics
        metrics = {
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "loss": loss.item()
        }
        
        if return_outputs:
            return loss, metrics
        
        return loss
    
    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """Log metrics with GRPO-specific information."""
        # Add GRPO-specific metrics to logs
        if "loss" in logs:
            logs["grpo/loss"] = logs["loss"]
        
        # Add stored metrics
        for key, value in self._stored_metrics.items():
            if isinstance(value, list) and len(value) > 0:
                logs[f"grpo/{key}"] = sum(value) / len(value)
                self._stored_metrics[key] = []
        
        return super().log(logs, *args, **kwargs)