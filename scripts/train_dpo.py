"""DPO fine-tune the v1b chess LLM on preference pairs from scripts/eval.py export-dpo.

Reads JSONL produced by `eval.py export-dpo` (one DpoPair per line with fields
``prompt`` / ``moves_uci`` / ``model_side`` / ``chosen`` / ``rejected``). The
ChessTokenizer has no FEN vocabulary, so the FEN ``prompt`` field is unused at
training time — the model is conditioned on ``moves_uci`` prefixed with the
perspective token (``<WHITE>`` or ``<BLACK>``), exactly matching the format
produced by ``Game.tokenize`` at inference time.

DPO hyperparams shifted from finetune.py:
- LR 5e-7 (60x lower than the fine-tune) — DPO is sensitive to large updates;
  the reference model anchors the policy and a higher LR collapses the KL.
- beta 0.1 (TRL default) — KL strength.
- Single move completion → max_length stays small (~256 tokens covers any
  realistic game length plus the one-move chosen/rejected suffix).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import torch
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from halluci_mate.chess_tokenizer import BLACK_TOKEN, WHITE_TOKEN, ChessTokenizer
from halluci_mate.eval.records import Side

load_dotenv()

# Side string → perspective token. Keys are anchored to the ``Side`` enum
# values the JSONL exporter writes, so a rename there surfaces as a typed
# break rather than a silent KeyError at training time. We deliberately do
# not import ``halluci_mate.game.Perspective``: it is an ``IntEnum`` of
# token *ids* (not the string tokens this script needs) and would pull
# ``chess.Board`` into the trainer's import graph.
_SIDE_TO_PERSPECTIVE_TOKEN: dict[str, str] = {Side.WHITE.value: WHITE_TOKEN, Side.BLACK.value: BLACK_TOKEN}


def _format_prompt(moves_uci: list[str], model_side: str) -> str:
    """Build the whitespace-tokenized prompt string the ChessTokenizer expects.

    The ChessTokenizer's ``_tokenize`` splits on whitespace, so a string of the
    form ``"<WHITE> m1 m2 ..."`` round-trips to ``[<WHITE>_id, m1_id, m2_id, ...]``
    — the exact sequence produced by ``Game.tokenize`` at inference time.
    """
    perspective = _SIDE_TO_PERSPECTIVE_TOKEN[model_side]
    return " ".join([perspective, *moves_uci])


def _load_pairs(path: Path) -> Dataset:
    """Load the export-dpo JSONL into a HF Dataset with prompt/chosen/rejected columns.

    A trailing space is appended to ``prompt`` so that ``prompt + chosen``
    re-splits into the original token list when DPOTrainer concatenates them
    for the chosen/rejected forward pass. Without it, the last history move
    and the labeled move would fuse into one whitespace token.
    """
    prompts: list[str] = []
    chosen: list[str] = []
    rejected: list[str] = []
    with path.open(encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            pair = json.loads(line)
            prompts.append(_format_prompt(pair["moves_uci"], pair["model_side"]) + " ")
            chosen.append(pair["chosen"])
            rejected.append(pair["rejected"])
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def main(
    pairs_path: Path = Path("dpo/v1b-consequential.jsonl"),
    output_directory: Path = Path("runs-v1b-dpo"),
    base_model: str = "jspaulsen/halluci-mate-v1b",
    beta: float = 0.1,
    learning_rate: float = 5e-7,
    epochs: int = 1,
    per_device_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.0,
    max_length: int = 256,
    eval_fraction: float = 0.02,
    seed: int = 4042,
) -> None:
    tokenizer = ChessTokenizer()

    dataset = _load_pairs(pairs_path)
    if eval_fraction > 0:
        split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = dataset, None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )

    state = PartialState()
    name_list: list[str | None] = [None]

    if state.is_main_process:
        mlflow.set_experiment("halluci-mate-v1b-dpo")
        mlflow.start_run()
        mlflow.log_params(
            {
                "base_model": base_model,
                "pairs_path": str(pairs_path),
                "beta": beta,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "per_device_batch_size": per_device_batch_size,
                "grad_accum_steps": gradient_accumulation_steps,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "max_length": max_length,
                "lr_scheduler_type": "cosine",
                "n_pairs": len(dataset),
                "seed": seed,
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

    config = DPOConfig(
        output_dir=str(output_directory),
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_length=max_length,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100,
        save_steps=200,
        save_total_limit=5,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        seed=seed,
        report_to="mlflow",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
