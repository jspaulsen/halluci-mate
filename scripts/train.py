"""Train a chess LLM from scratch using Qwen3-0.6B architecture."""

from pathlib import Path

import mlflow
import torch
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from halluci_mate.chess_tokenizer import ChessTokenizer

load_dotenv()


def main(
    warmup_ratio: float = 0.01,
    lr_scheduler: str = "cosine",
    output_directory: Path = Path("runs-v1"),
) -> None:
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.00  # Turn this on if overfitting

    model_path: str = "Qwen/Qwen3-0.6B"
    tokenizer = ChessTokenizer()

    # Load architecture only (training from scratch, not using pretrained weights)
    config = AutoConfig.from_pretrained(model_path)
    config.vocab_size = len(tokenizer)

    # Don't manually move to device - Trainer/accelerate handles device placement for multi-GPU
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )

    # Load pre-tokenized Parquet files from prepare_data.py
    dataset: Dataset = load_dataset("parquet", data_files="data/train.parquet", split="train")
    eval_dataset: Dataset = load_dataset("parquet", data_files="data/eval.parquet", split="train")

    # Using DataCollatorForLanguageModeling (not Seq2Seq) because we're doing pure causal LM
    # where every token predicts the next - no distinct input/output separation like chat models
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Only initialize mlflow on the main process to avoid duplicate runs
    state = PartialState()
    name_list = [None]

    if state.is_main_process:
        mlflow.set_experiment("halluci-mate")
        mlflow.start_run()
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler,
        })

        active_run = mlflow.active_run()
        if not active_run:
            raise RuntimeError("MLflow run failed to start")
        name_list[0] = active_run.info.run_name

    # 1. Single barrier to ensure the main process has finished setting up MLflow
    state.wait_for_everyone()

    # 2. Safely broadcast the run name from process 0 to all other processes
    name_list = broadcast_object_list(name_list, from_process=0)
    name = name_list[0] if name_list[0] is not None else "unknown"

    output_directory = output_directory / name
    output_directory.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            metric_for_best_model="loss",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=100,
            save_total_limit=10,
            # When pretraining, AdamW is usually better than paged optimizers which can introduce instability early on.
            # optim="paged_adamw_8bit",
            optim="adamw_torch_fused",
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler,
            max_grad_norm=1.0,
            seed=4042,
            output_dir=str(output_directory),
            report_to="mlflow",
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()
