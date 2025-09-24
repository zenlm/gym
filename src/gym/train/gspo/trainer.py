# Copyright 2025 Zoo Labs Foundation Inc. and the Gym team.
#
# Group Sequence Policy Optimization (GSPO) Trainer
# Based on Alibaba's GSPO: https://arxiv.org/abs/2507.18071
# Used in Qwen3 model training
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


class GSPOTrainer(Trainer):
    """
    Group Sequence Policy Optimization (GSPO) Trainer
    
    GSPO performs sequence-level optimization rather than token-level,
    providing better training stability especially for MoE models.
    It uses theoretically grounded importance ratios derived from sequence likelihood.
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
        
        # GSPO specific parameters
        self.group_size = getattr(finetuning_args, "gspo_group_size", 8)
        self.beta = getattr(finetuning_args, "gspo_beta", 0.1)
        self.clip_epsilon = getattr(finetuning_args, "gspo_clip_epsilon", 0.2)
        self.sequence_level = getattr(finetuning_args, "gspo_sequence_level", True)
        self.normalize_rewards = getattr(finetuning_args, "gspo_normalize_rewards", True)
        self.moe_stabilization = getattr(finetuning_args, "gspo_moe_stabilization", True)
        
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
        
        # Configure for stable training
        if hasattr(model, "config"):
            model.config.use_cache = False
            # Disable dropout for stable training
            if hasattr(model.config, "dropout"):
                model.config.dropout = 0.0
            # Enable MoE stabilization if model has experts
            if self.moe_stabilization and hasattr(model.config, "num_experts"):
                logger.info_rank0(f"MoE model detected with {model.config.num_experts} experts. Enabling stabilization.")
        
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
    
    def compute_sequence_likelihood(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sequence-level likelihood for GSPO.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target token IDs [batch_size, seq_len]
            mask: Optional mask for valid tokens [batch_size, seq_len]
            
        Returns:
            sequence_likelihood: Sequence-level log probabilities [batch_size]
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for target tokens
        batch_size, seq_len = labels.shape
        token_log_probs = torch.gather(
            log_probs.view(-1, log_probs.size(-1)),
            dim=1,
            index=labels.view(-1, 1)
        ).view(batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            token_log_probs = token_log_probs * mask
            sequence_likelihood = token_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            sequence_likelihood = token_log_probs.mean(dim=1)
        
        return sequence_likelihood
    
    def compute_importance_ratio(
        self,
        policy_likelihood: torch.Tensor,
        reference_likelihood: torch.Tensor,
        clip: bool = True
    ) -> torch.Tensor:
        """
        Compute sequence-level importance ratio for GSPO.
        
        Args:
            policy_likelihood: Policy model sequence likelihoods
            reference_likelihood: Reference model sequence likelihoods
            clip: Whether to apply clipping
            
        Returns:
            importance_ratio: Clipped importance ratios
        """
        # Compute log ratio
        log_ratio = policy_likelihood - reference_likelihood
        ratio = torch.exp(log_ratio)
        
        # Apply clipping if requested
        if clip:
            ratio = torch.clamp(
                ratio,
                1 - self.clip_epsilon,
                1 + self.clip_epsilon
            )
        
        return ratio
    
    def compute_group_rewards(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Normalize rewards within groups for better optimization.
        
        Args:
            rewards: Raw rewards [batch_size]
            group_size: Size of each group
            
        Returns:
            normalized_rewards: Group-normalized rewards [batch_size]
        """
        if group_size is None:
            group_size = self.group_size
        
        batch_size = rewards.shape[0]
        num_groups = math.ceil(batch_size / group_size)
        
        normalized = torch.zeros_like(rewards)
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, batch_size)
            
            group_rewards = rewards[start_idx:end_idx]
            
            if self.normalize_rewards:
                # Normalize within group
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                normalized[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            else:
                normalized[start_idx:end_idx] = group_rewards
        
        return normalized
    
    def stabilize_moe_routing(
        self,
        model_outputs: Any,
        expert_masks: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Stabilize MoE expert routing for consistent training.
        
        Args:
            model_outputs: Model forward outputs
            expert_masks: Optional expert selection masks
            
        Returns:
            metrics: MoE stabilization metrics
        """
        metrics = {}
        
        if hasattr(model_outputs, "router_logits"):
            # Track expert utilization
            router_logits = model_outputs.router_logits
            expert_probs = F.softmax(router_logits, dim=-1)
            
            # Compute load balancing loss
            expert_usage = expert_probs.mean(dim=0).mean(dim=0)
            load_balance_loss = expert_usage.var() * 0.01  # Small regularization
            
            metrics["moe/load_balance_loss"] = load_balance_loss.item()
            metrics["moe/expert_usage_std"] = expert_usage.std().item()
            
            # Add auxiliary loss to stabilize routing
            if self.training:
                return load_balance_loss
        
        return metrics
    
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
        Compute GSPO loss for a batch of inputs.
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate responses
        responses, response_mask = self.generate_responses(input_ids, attention_mask)
        
        # Compute rewards
        rewards = self.compute_rewards(input_ids, responses)
        
        # Normalize rewards within groups
        normalized_rewards = self.compute_group_rewards(rewards)
        
        # Concatenate inputs and responses
        full_input_ids = torch.cat([input_ids, responses], dim=1)
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)
        
        # Create labels (shift by 1 for next token prediction)
        labels = full_input_ids.clone()
        labels[:, :-1] = full_input_ids[:, 1:]
        labels[:, -1] = self.tokenizer.pad_token_id
        
        # Mask out prompt tokens from labels
        prompt_length = input_ids.shape[1]
        labels[:, :prompt_length] = -100  # Ignore prompt in loss
        
        # Get policy model outputs
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            policy_outputs = model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                return_dict=True
            )
            policy_logits = policy_outputs.logits
        
        # Compute sequence likelihood for policy
        policy_likelihood = self.compute_sequence_likelihood(
            policy_logits[:, :-1, :],  # Shift for next token prediction
            labels[:, 1:],  # Target tokens
            full_attention_mask[:, 1:]  # Mask
        )
        
        # Get reference model likelihood
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits
                
                ref_likelihood = self.compute_sequence_likelihood(
                    ref_logits[:, :-1, :],
                    labels[:, 1:],
                    full_attention_mask[:, 1:]
                )
        else:
            # If no reference model, use detached policy likelihood
            ref_likelihood = policy_likelihood.detach()
        
        # Compute importance ratio
        importance_ratio = self.compute_importance_ratio(
            policy_likelihood,
            ref_likelihood,
            clip=True
        )
        
        # Compute GSPO loss (sequence-level policy gradient)
        advantages = normalized_rewards
        gspo_loss = -(importance_ratio * advantages).mean()
        
        # Add MoE stabilization if applicable
        moe_metrics = {}
        if self.moe_stabilization and hasattr(policy_outputs, "router_logits"):
            moe_aux_loss = self.stabilize_moe_routing(policy_outputs)
            if isinstance(moe_aux_loss, torch.Tensor):
                gspo_loss = gspo_loss + moe_aux_loss
                moe_metrics = {"moe/auxiliary_loss": moe_aux_loss.item()}
        
        # Collect metrics
        metrics = {
            "rewards_mean": rewards.mean().item(),
            "rewards_std": rewards.std().item(),
            "normalized_rewards_mean": normalized_rewards.mean().item(),
            "importance_ratio_mean": importance_ratio.mean().item(),
            "policy_likelihood": policy_likelihood.mean().item(),
            "ref_likelihood": ref_likelihood.mean().item(),
            "loss": gspo_loss.item(),
            **moe_metrics
        }
        
        if return_outputs:
            return gspo_loss, metrics
        
        return gspo_loss
    
    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """Log metrics with GSPO-specific information."""
        # Add GSPO-specific metrics to logs
        if "loss" in logs:
            logs["gspo/loss"] = logs["loss"]
        
        # Add stored metrics
        for key, value in self._stored_metrics.items():
            if isinstance(value, list) and len(value) > 0:
                logs[f"gspo/{key}"] = sum(value) / len(value)
                self._stored_metrics[key] = []
        
        return super().log(logs, *args, **kwargs)