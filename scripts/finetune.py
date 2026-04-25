"""Fine-tune the v1a chess LLM on a high-Elo subset.

Loads pretrained weights from jspaulsen/halluci-mate-v1a (NOT from_config —
this is a fine-tune, not from-scratch training) and continues training on
the high-Elo dataset produced by scripts/prepare_finetune_data.py.

Hyperparams shifted from train.py:
- LR 3e-5 (10x lower than v1a) — standard fine-tune practice, weights are warm
- Warmup 0.005 (smaller) — model is already converged
"""

from pathlib import Path

import mlflow
import torch
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from halluci_mate.chess_tokenizer import ChessTokenizer

load_dotenv()


def main(
    warmup_ratio: float = 0.005,
    output_directory: Path = Path("runs-v1a-ft"),
) -> None:
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    learning_rate: float = 3e-5  # 10x lower than v1a pretrain — fine-tune from warm weights
    weight_decay: float = 0.01
    base_model: str = "jspaulsen/halluci-mate-v1a"
    tokenizer = ChessTokenizer()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )

    dataset: Dataset = load_dataset("parquet", data_files="data/v1a-highelo/train.parquet", split="train")
    eval_dataset: Dataset = load_dataset("parquet", data_files="data/v1a-highelo/eval.parquet", split="train")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    state = PartialState()
    name_list = [None]

    if state.is_main_process:
        mlflow.set_experiment("halluci-mate-v1a-finetune")
        mlflow.start_run()
        mlflow.log_params(
            {
                "base_model": base_model,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum_steps": gradient_accumulation_steps,
                "warmup_ratio": warmup_ratio,
                "lr_scheduler_type": "cosine_with_min_lr",
            }
        )

        active_run = mlflow.active_run()
        if not active_run:
            raise RuntimeError("MLflow run failed to start")
        name_list[0] = active_run.info.run_name

    state.wait_for_everyone()

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
            optim="adamw_torch_fused",
            weight_decay=weight_decay,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr": learning_rate * 0.1},
            max_grad_norm=1.0,
            dataloader_num_workers=4,
            seed=4042,
            output_dir=str(output_directory),
            report_to="mlflow",
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()
