"""Train a chess LLM from scratch using Qwen3-0.6B architecture.

TODO: Load pre-built Parquet files from prepare_data.py instead of raw streaming.
"""

from pathlib import Path

import torch
import wandb
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
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.00  # Turn this on if overfitting

    model_path: str = "Qwen/Qwen3-0.6B"
    tokenizer = ChessTokenizer()

    # Load architecture only (training from scratch, not using pretrained weights)
    config = AutoConfig.from_pretrained(model_path)
    config.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")

    # TODO: Load pre-built Parquet files from prepare_data.py output
    dataset: Dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)  # type: ignore
    dataset = dataset.filter(lambda x: x["Termination"] == "Normal")

    # Using DataCollatorForLanguageModeling (not Seq2Seq) because we're doing pure causal LM
    # where every token predicts the next - no distinct input/output separation like chat models
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    wandb.init(
        project="halluci-mate",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler,
        },
    )

    if not wandb.run or not wandb.run.name:
        raise ValueError("WandB run name is not set. Please set it before running the script.")

    name = wandb.run.name
    output_directory = output_directory / name
    output_directory.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
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
            save_steps=100,
            save_total_limit=10,
            optim="paged_adamw_8bit",
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler,
            max_grad_norm=1.0,
            seed=4042,
            output_dir=str(output_directory),
            report_to="wandb",
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()
