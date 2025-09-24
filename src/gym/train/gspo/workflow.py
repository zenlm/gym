# Copyright 2025 Zoo Labs Foundation Inc. and the Gym team.
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

from typing import TYPE_CHECKING, Optional
import torch
from transformers import DataCollatorForSeq2Seq, TrainingArguments

from ...data import get_dataset, get_template_and_fix_tokenizer, split_dataset
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...model import load_model, load_tokenizer
from ..callbacks import LogCallback
from .trainer import GSPOTrainer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = logging.get_logger(__name__)


def run_gspo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
) -> None:
    """
    Run Group Sequence Policy Optimization training.
    
    GSPO is particularly effective for training large MoE models,
    providing better stability than token-level methods like PPO or GRPO.
    
    Args:
        model_args: Arguments for model configuration
        data_args: Arguments for data configuration
        training_args: Arguments for training configuration
        finetuning_args: Arguments for fine-tuning configuration
        generating_args: Arguments for generation configuration
        callbacks: Optional list of trainer callbacks
    """
    # Load tokenizer and configure template
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    
    # Configure template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Load dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = get_dataset(template, model_args, data_args, training_args, "gspo", tokenizer)
        
        # Split dataset if needed
        if training_args.do_eval:
            if data_args.val_size > 0:
                dataset = split_dataset(dataset, data_args, training_args.seed)
                train_dataset = dataset["train"]
                eval_dataset = dataset["eval"]
            else:
                train_dataset = dataset
                eval_dataset = None
        else:
            train_dataset = dataset
            eval_dataset = None
    
    # Log dataset info
    if train_dataset is not None:
        logger.info_rank0(f"Training samples: {len(train_dataset):,}")
        logger.info_rank0(f"Training example: {train_dataset[0]}")
    
    if eval_dataset is not None:
        logger.info_rank0(f"Evaluation samples: {len(eval_dataset):,}")
    
    # Load model
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        training_args.do_train,
        add_value_head=False  # GSPO doesn't use value head
    )
    
    # Check if model is MoE
    is_moe = False
    if hasattr(model, "config") and hasattr(model.config, "num_experts"):
        is_moe = True
        logger.info_rank0(f"MoE model detected with {model.config.num_experts} experts")
        logger.info_rank0("GSPO is particularly effective for MoE training")
    
    # Load reference model if needed
    ref_model = None
    if finetuning_args.use_ref_model:
        logger.info_rank0("Loading reference model...")
        ref_model = load_model(
            tokenizer,
            model_args,
            finetuning_args,
            False,  # Don't train reference model
            add_value_head=False
        )
        ref_model.eval()
        # Disable dropout in reference model
        for module in ref_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        label_pad_token_id=IGNORE_INDEX
    )
    
    # Initialize callbacks
    if callbacks is None:
        callbacks = []
    
    callbacks.append(LogCallback())
    
    # Initialize trainer
    trainer = GSPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        model_args=model_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Training
    if training_args.do_train:
        logger.info_rank0("*** Starting GSPO training ***")
        logger.info_rank0(f"  Num examples = {len(train_dataset):,}")
        logger.info_rank0(f"  Num epochs = {training_args.num_train_epochs}")
        logger.info_rank0(f"  Group size = {finetuning_args.gspo_group_size}")
        logger.info_rank0(f"  Beta = {finetuning_args.gspo_beta}")
        logger.info_rank0(f"  Clip epsilon = {finetuning_args.gspo_clip_epsilon}")
        logger.info_rank0(f"  Sequence-level optimization = {finetuning_args.gspo_sequence_level}")
        if is_moe:
            logger.info_rank0(f"  MoE stabilization = {finetuning_args.gspo_moe_stabilization}")
        
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval and eval_dataset is not None:
        logger.info_rank0("*** Evaluating ***")
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Create model card
    if training_args.push_to_hub:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "text-generation",
            "dataset": data_args.dataset,
            "tags": ["gspo", "reinforcement-learning", "gym", "qwen3"],
        }
        if is_moe:
            kwargs["tags"].append("moe")
        trainer.push_to_hub(**kwargs)