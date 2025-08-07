"""
ASIDE Fine-tuning Script

This is the main training script for reproducing the ASIDE (Architecturally Separated 
Instruction-Data Embeddings) experiments. It implements the supervised fine-tuning (SFT) 
procedure described in Section 4.1 of the paper using the Alpaca dataset.

Key Features:
- Trains models with ASIDE architectural modifications (rotation-based instruction/data separation)
- Supports multiple embedding types: single_emb , double_emb, ISE, forward_rot (=ASIDE)
- Implements gradual rotation during training (experimental feature)
- Distributed training support with DeepSpeed integration

Usage:
     srun --export=ALL deepspeed --num_gpus=8 --master_port=29509 \
        fine-tune.py --model_family llama_2_13b --train_version SFTv110 \
        --emb_type forward_rot --model_ix 1 --run_number 0 \
        --train_type full --num_train_epochs 3 --per_device_train_batch_size 2 \  
        --gradient_accumulation_steps 4 --learning_rate 1e-6 --lr_scheduler_type cosine \ 
        --warmup_ratio 0 --logging_steps 10 --evaluation_strategy epoch \ 
        --save_strategy epoch --eval_steps 1 --save_steps 1 --save_total_limit 1 \ 
        --load_best_model_at_end True --prediction_loss_only True --bf16 True \ 
        --embedding_init rot_isoclinic --rotation_alpha 1.57079633 \ 
        --learned_rotation False --add_linear_shift False --rotation_direction right \
        --gradual_rotation False


References:
    ASIDE: Architectural Separation of Instructions and Data in Language Models
    Section 4.1: Training procedure
"""

import os
import json
import torch

from typing import Any, Dict, List

from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from transformers import TrainerCallback, TrainerState, TrainerControl

import math
from transformers.trainer_utils import is_main_process


import argparse
import logging

from trl import SFTTrainer
from model import *
from typing import List, Dict
from torch.utils.data import Dataset

from model_api import *
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from embeddings_init import generate_isoclinic_rotation_matrix



def merge_zero_shards_and_save(checkpoint_dir):
    """
    Merges DeepSpeed ZeRO-sharded checkpoints into a single FP32 state dict.
    
    This function converts distributed DeepSpeed checkpoints (which are sharded across
    multiple GPUs) into a single consolidated checkpoint file that can be loaded on
    any device. Essential for saving final model checkpoints after distributed training.
    
    Args:
        checkpoint_dir (str): Directory containing the ZeRO-sharded checkpoint files.
                             Typically contains files like zero_pp_rank_*_mp_rank_00_model_states.pt
        
    """
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=checkpoint_dir,
        output_dir=checkpoint_dir,
    )

    print(f"Saved merged FP32 checkpoint to {checkpoint_dir}")




class AlpacaDataset(Dataset):
    """
    Custom Dataset class for ASIDE instruction-tuning with Alpaca data.
    
    This dataset provides the instruction-data separation on the level of 
    prompts required for ASIDE training.
    It formats Alpaca examples into distinct instruction and data segments, enabling
    the model to learn different representations for executable vs non-executable content.
    
    The dataset follows the paper's methodology (Section 4.1) by:
    1. Separating instruction and data components of each example
    2. Creating segment IDs to distinguish instruction from data tokens (needed for ASIDE and ISE)
    3. Preparing inputs compatible with ASIDE's conditional embedding mechanism
    
    Args:
        data (List[Dict]): Alpaca dataset examples with 'instruction', 'input', 'output' fields
        template_info (Dict): Chat template information for formatting prompts
        handler (CustomModelHandler): Model handler containing tokenizer and embedding configuration
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        
    Note:
        The dataset ensures that instruction tokens receive standard embeddings while
        data tokens receive rotated embeddings during ASIDE forward passes.
    """

    def __init__(self, data: List[Dict], template_info: Dict, handler: CustomModelHandler, max_length=512):
        self.data = data
        self.template_info = template_info
        self.handler = handler
        self.max_length = max_length
        self.printed_input = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Prepares a single training example with instruction/data separation.
        
        Returns:
            Dict: Contains 'input_ids', 'attention_mask', and optionally 'segment_ids'
                  for ASIDE models. segment_ids indicate whether each token is part
                  of instructions (0) or data (1).
        """
        data_point = self.data[idx]
        # Format the prompts using your existing function
        template = self.template_info["template_prompt"]
        instruction = format_prompt(data_point["instruction"], template, "system")
        data = format_prompt(data_point["input"], template, "user")
        output = format_prompt(data_point["output"], template, "output")

        # Prepare text sequences
            
        text_sequences = format_model_input(self.handler.tokenizer, instruction, data, output, split_chat=self.handler.split_chat)

        # Get input_ids and attention_mask
        input_ids, attention_mask, segment_ids = texts_to_prepared_ids(
            text_sequences, self.handler.tokenizer, self.max_length, self.handler.embedding_type 
        )
        input_ids = torch.hstack((input_ids, torch.Tensor([[self.handler.tokenizer.eos_token_id]]))).long()
        attention_mask = torch.hstack((attention_mask, torch.Tensor([[True]]))).bool()
        if segment_ids is not None:
            segment_ids = torch.hstack((segment_ids, torch.Tensor([[(1 + len(text_sequences)) % 2]]))).long() 

        if not self.printed_input: 
            print("Input Text Sequence:\n", text_sequences, input_ids, segment_ids, attention_mask)
            #print("Shapes", input_ids.shape, segment_ids.shape, attention_mask.shape)
            self.printed_input = True
        if segment_ids is not None:
            return {
                "input_ids": input_ids.squeeze(0).tolist(), 
                "attention_mask": attention_mask.squeeze(0).tolist(),
                "segment_ids": segment_ids.squeeze(0).tolist()
            }
        return {
                "input_ids": input_ids.squeeze(0).tolist(), 
                "attention_mask": attention_mask.squeeze(0).tolist(),
            }
    
    def map(self, function, batched: bool = False, batch_size: int = None, **kwargs):
        """
        Applies a function to dataset elements (needed for HuggingFace datasets compatibility).
        
        Args:
            function: Function to apply to each element or batch
            batched (bool): Whether to process in batches
            batch_size (int, optional): Batch size if batched=True
            **kwargs: Additional arguments passed to function
            
        Returns:
            AlpacaDataset: New dataset with function applied
        """
        new_data = []
        if batched:
            batch_size = batch_size or 1
            for i in range(0, len(self.data), batch_size):
                batch = self.data[i : i + batch_size]
                # If your function is meant to handle a batch, it should return a list.
                result = function(batch, **kwargs)
                # If result is not a list, wrap it in a list.
                if not isinstance(result, list):
                    result = [result]
                new_data.extend(result)
        else:
            for item in self.data:
                new_data.append(function(item, **kwargs))
        return AlpacaDataset(new_data, template_info=self.template_info, handler=self.handler, max_length=self.max_length)



class CustomSFTTrainer(SFTTrainer):
    """
    Extended SFTTrainer with custom logging.
    
    This trainer extends the standard SFTTrainer to provide:
    - Detailed loss logging for analysis
    - Custom metrics tracking for ASIDE experiments
    
    Args:
        loss_log_file (str, optional): Path to save detailed training logs in JSON format
        *args, **kwargs: Arguments passed to parent SFTTrainer
        
    """
    def __init__(self, *args, loss_log_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log_file = loss_log_file
        self.loss_logs = {"train_loss": [], "eval_loss": [], "metrics": [], "steps": [], "eval_steps": []}

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().on_step_end(args, state, control, **kwargs)

    def log(self, logs, *args, **kwargs):
        """
        Logging with local JSON storage.
        
        Captures training/evaluation losses and metrics, saving them to a JSON file
        for detailed analysis of ASIDE training progress.
        """
        super().log(logs, *args, **kwargs)  # Call the original log method
        if (self.loss_log_file is not None and dist.get_rank() == 0):  # Ensure only rank 0 process logs
            # Track and log training loss
            train_loss = logs.get("loss", None)
            if train_loss is not None:
                logging.info(f"Step {self.state.global_step}: Train Loss = {train_loss:.4f}")
                self.loss_logs["train_loss"].append(train_loss)
                self.loss_logs["steps"].append(self.state.global_step)


            # Track evaluation loss
            eval_loss = logs.get("eval_loss", None)
            if eval_loss is not None:
                logging.info(f"Step {self.state.global_step}: Eval Loss = {eval_loss:.4f}")
                self.loss_logs["eval_steps"].append(self.state.global_step)
                self.loss_logs["eval_loss"].append(eval_loss)
            token_accuracy = logs.get("token_accuracy", None)
            if token_accuracy is not None:
                logging.info(f"Step {self.state.global_step}: Token Accuracy = {token_accuracy:.4f}")
                if "token_accuracy" not in self.loss_logs:
                    self.loss_logs["token_accuracy"] = []
                self.loss_logs["token_accuracy"].append(token_accuracy)

            # Save logs to the JSON file after every log update
            with open(self.loss_log_file, "w") as f:
                json.dump(self.loss_logs, f, indent=4)






class CustomDataCollator:
    """
    Custom data collator for ASIDE models with segment ID support.
    
    This collator extends the standard language modeling collator to handle
    segment IDs that distinguish instruction from data tokens. Essential for
    ASIDE's conditional embedding mechanism.
    
    Args:
        tokenizer: HuggingFace tokenizer
        mlm (bool): Whether to use masked language modeling (typically False for ASIDE)
        
    Note:
        Segment IDs are crucial for ASIDE as they determine which tokens receive
        rotated embeddings during the forward pass.
    """
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    def __call__(self, features):
        """
        Collates batch with segment ID handling.
        
        Returns:
            Dict: Batch dictionary with 'input_ids', 'attention_mask', 'labels',
                  and 'segment_ids' (for ASIDE and ISE models)
        """
        if "segment_ids" in features[0] and features[0]["segment_ids"] is not None:
            # Collect segment_ids from each feature and convert them to tensors.
            segment_ids = [torch.tensor(feature.pop("segment_ids")) for feature in features]
            padded_segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
            batch = self.base_collator(features)
            batch["segment_ids"] = padded_segment_ids
        else: 
            batch = self.base_collator(features)
        return batch 




class EmbeddingRotationCallback(TrainerCallback):
    """
    Callback for gradual rotation during ASIDE training (experimental feature).
    
    This callback implements gradual rotation where the rotation angle increases
    linearly from 0 to the target angle during the first epoch. This experimental
    approach allows the model to gradually adapt to the architectural changes.
    
    Args:
        total_steps_per_epoch (int): Total training steps in one epoch
        
    Note:
        This is an experimental feature mentioned in the paper as future work.
        The standard ASIDE method applies full rotation (π/2) from the beginning.
        
    Mathematical Implementation:
        rotation_alpha(step) = target_alpha * (step / total_steps_per_epoch)
        where step ∈ [0, total_steps_per_epoch]
    """
    def __init__(self, total_steps_per_epoch: int):
        self.total_steps_per_epoch = total_steps_per_epoch

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        # Determine the current step within the epoch.
        if state.global_step <= self.total_steps_per_epoch:
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            dim = model.config.hidden_size
            # Calculate the rotation alpha based on the current step.
            rotation_alpha = model.global_rotation_alpha * (state.global_step / self.total_steps_per_epoch)
            
            with torch.no_grad():
                if dist.get_rank() == 0:
                    # create the new matrix
                    new_matrix = generate_isoclinic_rotation_matrix(dim, rotation_alpha, device, model_dtype)
                else:
                    # create a placeholder for it
                    new_matrix = torch.empty_like(model.rotation_matrix)

                dist.broadcast(new_matrix, src=0)
                model.rotation_matrix.data.copy_(new_matrix)
            
            # Broadcast the computed value (from rank 0) to all processes.

            
            if dist.get_rank() == 0:
                print(f"On step {state.global_step} set embedding rotation angle to {rotation_alpha}")
            
        return control



def main(model_family: str, emb_type: str, train_version: str, model_ix: int, run_number: int, hparams: Dict[str, Any]):
    """
    Main training function for ASIDE experiments.
    
    1. Loads configuration and datasets
    2. Initializes ASIDE model with appropriate embedding type
    3. Sets up custom trainer with logging
    4. Executes supervised fine-tuning on Alpaca data
    5. Saves final checkpoint with metadata
    
    Args:
        model_family (str): Model family identifier (e.g., "llama_3.1_8b")
        emb_type (str): Embedding type - determines ASIDE variant:
                       - "single_emb": Vanilla model
                       - "double_emb": Legacy double embedding approach
                       - "ise": ISE baseline method
                       - "forward_rot": Main ASIDE method with isoclinic rotation
        train_version (str): Training version identifier (e.g., "SFTv110")
        model_ix (int): Index of model in configuration's pure_models list
        run_number (int): Run number for experiment tracking
        hparams (Dict[str, Any]): Training hyperparameters
        
    """
    config_path = f"./configs/config_{model_family}_{train_version}.json"
    config = load_config(config_path)

    if dist.get_rank() == 0:
        print(config)
        print("\n", config["models"][emb_type]["pure_models"])

    pure_model_info = config["models"][emb_type]["pure_models"][model_ix]
    checkpoint_to_load_to = pure_model_info["checkpoint_to_load_to"]
    checkpoint_to_load_from = pure_model_info["checkpoint_to_load_from"]
    tokenizer_path = config["tokenizer_path"]
    instruct_model_path = pure_model_info.get("instruct_model_path", None)
    data_model_path = pure_model_info.get("data_model_path", None)
    chat_template_path = pure_model_info["chat_template_path"]

    train_dataset_path = config["train_dataset_path"]
    assert "eval_dataset_path" in config.keys(), "Update Config to new format"
    eval_dataset_path = config["eval_dataset_path"]
    train_version = config["training_version"]
    model_type = "from_inst" if (model_ix == 0) else "from_base"
    output_dir = checkpoint_to_load_to + f"/train_checkpoints/{train_version}/{model_type}_run_{run_number}"

    # Include hyperparams in run name
    run_name = f"{train_version}/{pure_model_info['name']}_train_{hparams['train_type']}_{train_version}_run={run_number}_alpha={hparams['rotation_alpha']}"

    train_logs_path = os.path.join(config["train_logs_path"], model_family, run_name)
    os.makedirs(train_logs_path, exist_ok=True)

    log_file = os.path.join(train_logs_path, "losses.log")  # Specify file path
    log_file_json = os.path.join(train_logs_path, "losses_metrics.json")  # Specify file path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Save to file
            logging.StreamHandler()  # Print to console
        ]
    )

    # Load dataset
    train_data = load_dataset("json", data_files=train_dataset_path)["train"]
    eval_data = load_dataset("json", data_files=eval_dataset_path)["train"]
    assert len(train_data) == 46581
    assert len(eval_data) == 5175
    with open(config["prompt_templates_path"], "r") as f:
        templates = json.load(f)
    template_info = {
        "template_prompt": templates[0]
    }

    handler = CustomModelHandler(
        checkpoint_to_load_from, 
        instruct_model_path, 
        data_model_path, 
        tokenizer_path, 
        chat_template_path, 
        embedding_type=emb_type,
        embeddings_init=hparams["embedding_init"],
        rotation_alpha=hparams["rotation_alpha"],
        add_linear_shift=hparams["add_linear_shift"],
        rotation_direction=hparams["rotation_direction"], 
        learned_rotation=hparams["learned_rotation"],
        gradual_rotation=hparams["gradual_rotation"],
        model_dtype=torch.bfloat16,
        load_from_checkpoint=checkpoint_to_load_from is not None,
        post_init_rotation=hparams["post_init_rotation"]
    )

    if handler.tokenizer.pad_token is None:
        print("WARNING: pad_token is None")
    max_length = hparams["max_length"]
    train_dataset = AlpacaDataset(train_data, template_info, handler, max_length=max_length)
    eval_dataset = AlpacaDataset(eval_data, template_info, handler, max_length=max_length)
    data_collator = CustomDataCollator(
        tokenizer=handler.tokenizer,
        mlm=False
    )
    
    if handler.tokenizer.pad_token is None:
        print("WARNING: pad_token is None")

    print('Datasets created')
 
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hparams["num_train_epochs"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        per_device_eval_batch_size=hparams["per_device_eval_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        warmup_ratio=hparams["warmup_ratio"],
        logging_dir=train_logs_path,
        logging_steps=hparams["logging_steps"],
        log_level="info",
        eval_strategy=hparams["evaluation_strategy"],
        save_strategy=hparams["save_strategy"],
        eval_steps=hparams["eval_steps"],
        save_steps=hparams["save_steps"],
        save_total_limit=hparams["save_total_limit"],
        load_best_model_at_end=hparams["load_best_model_at_end"],
        prediction_loss_only=hparams["prediction_loss_only"],
        bf16=hparams["bf16"],
        remove_unused_columns=hparams["remove_unused_columns"],
        deepspeed="deepspeed_config.json", 
        report_to=hparams["report_to"],
        metric_for_best_model=None,
        gradient_checkpointing=True,  
    )
    handler.model.config.use_cache = False
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    total_steps_per_epoch = math.ceil(
    len(train_dataset) / (world_size * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
)
    print(f"Total steps per epoch: {total_steps_per_epoch}")
    callbacks = [EmbeddingRotationCallback(total_steps_per_epoch)] if hparams["gradual_rotation"] else []

    trainer = CustomSFTTrainer(
        model=handler.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=handler.tokenizer,
        data_collator=data_collator,
        loss_log_file=log_file_json,
        callbacks=callbacks,
    )
    print('Trainer created')
    
    # Start training
    trainer.train()

    # Save the trained model and tokenizer
    print("Custom impl., saving last checkpoint")
    final_checkpoint_path = os.path.join(output_dir, "last")
    trainer.save_model(final_checkpoint_path)
    merge_zero_shards_and_save(
        checkpoint_dir=final_checkpoint_path
    )
    # Update config with new checkpoint info
    new_checkpoint_info = {
        "desc": f"{pure_model_info['desc']} trained with {train_version}",
        "name": f"{pure_model_info['name']}_{train_version}",
        "checkpoint_path": final_checkpoint_path,
        "instruct_model_path": instruct_model_path,
        "data_model_path": instruct_model_path,
        "chat_template_path": chat_template_path,
        "parent_pure_model_name": pure_model_info['name'],
        "run_number": run_number,
        "hyperparams": hparams
    }

    run_info_file = os.path.join(train_logs_path, "run_info.json")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_main_process(local_rank):
        with open(run_info_file, "w+") as f:
            json.dump(new_checkpoint_info, f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with optional hyperparameters.")
    parser.add_argument("--model_family", type=str, help="E.g., tiny_old, llama3_8b, etc.")
    parser.add_argument("--emb_type", type=str, choices=["double_emb", "single_emb",  "ise", "forward_rot"], help="Embedding Type")
    parser.add_argument("--model_ix", type=int, help="Index of the model in the pure_models list.")
    parser.add_argument("--run_number", type=str, default=0, help="Number of the run.")
    parser.add_argument("--train_version", type=str, help="e.g. SFTv11")

    parser.add_argument("--train_type", type=str, default="full", help="full or lora")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints and logs.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="device eval batch size")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--max_length", type=int, default=768, help="Max length of dataset")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine"],
                        help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Steps between each logging.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["steps", "epoch"],
                        help="Evaluation strategy.")
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"],
                        help="Save strategy.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluations.")
    parser.add_argument("--save_steps", type=int, default=300, help="Number of steps between model checkpoints.")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Maximum number of checkpoints to save.")
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=False,
                        help="Whether to load the best model at the end of training.")
    parser.add_argument("--prediction_loss_only", type=str2bool, default=False,
                        help="If True, only the prediction loss is used.")
    parser.add_argument("--bf16", type=bool, default=True, help="Use bf16 precision if available.")
    parser.add_argument("--activation_checkpointing", type=bool, default=False,
                        help="Whether to use gradient checkpointing.")
    parser.add_argument("--remove_unused_columns", type=bool, default=False, help="Remove unused columns in dataset.")
    parser.add_argument("--report_to", type=list, default=["none"],
                        help="Reporting framework (e.g., wandb, tensorboard).")
    parser.add_argument("--embedding_init", type=str, default=None, choices=[None, "copy", "rot_ind", "rot_isoclinic"],
                        help="Embedding initialization.")
    parser.add_argument("--rotation_alpha", type=float, default=None,
                        help="Embedding rotation constant.")
    parser.add_argument("--add_linear_shift", type=str2bool, default=False,
                    help="Add linear shift before rotation.")
    parser.add_argument("--learned_rotation", type=str2bool, default=None,
                    help="If rotation is parameter.")
    parser.add_argument("--gradual_rotation", type=str2bool, default=None,
                    help="If rotation is gradual.")
    parser.add_argument("--rotation_direction", type=str, default="right",
                    help="Embedding rotation direction.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training on GPUs.")

    parser.add_argument("--post_init_rotation", type=str2bool, default=False, help="Rotate embedding after initialization (normally used when loading from checkpoint).")

    parser.add_argument("--gradual_rot", type=str2bool, default=False, help="Use gradual rotation every step of training during 1st epoch")

    # Parse the arguments
    args = parser.parse_args()

    # Prepare user_hparams
    user_hparams = vars(args)  # Convert parsed arguments into dictionary
    model_family = user_hparams.pop("model_family")
    emb_type = user_hparams.pop("emb_type")
    model_ix = user_hparams.pop("model_ix")
    run_number = user_hparams.pop("run_number")
    train_version = user_hparams.pop("train_version")
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    main(model_family, emb_type, train_version, model_ix, run_number, user_hparams)
