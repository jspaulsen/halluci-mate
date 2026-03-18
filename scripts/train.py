import os
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import wandb


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
    # TODO: Create custom tokenizer with ~1800 legal chess moves
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # placeholder

    # Load architecture only (training from scratch, not using pretrained weights)
    config = AutoConfig.from_pretrained(model_path)
    # TODO: config.vocab_size = len(chess_tokenizer)  # resize for chess vocab
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")

    # Load Lichess dataset and filter for normal terminations only
    # TODO: Limit dataset size - full dataset is ~7B rows, need to sample/stream
    dataset: Dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)  # type: ignore
    dataset = dataset.filter(lambda x: x["Termination"] == "Normal")

    # TODO: Restructure data for training - convert PGN moves to appropriate format
    # dataset = dataset.map(
    #     lambda sample: _process_chess_game(sample, tokenizer),
    #     remove_columns=[name for name in dataset.column_names if name not in ['input_ids', 'attention_mask', 'labels']],
    #     num_proc=16,
    # )

    # split = dataset.train_test_split(test_size=0.001)

    # dataset = split["train"]
    # eval_dataset = split["test"]

    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


    wandb.init(
        project="halluci-mate",
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler,
        }
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
        train_dataset=dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            # warmup_steps=500,

            metric_for_best_model="loss",
            eval_strategy="steps",
            eval_steps=100,

            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=100,
            save_total_limit=10,

            # https://huggingface.co/docs/bitsandbytes/v0.43.0/en/optimizers#paged-optimizers
            optim="paged_adamw_8bit",
            # optim="adamw_8bit"
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
