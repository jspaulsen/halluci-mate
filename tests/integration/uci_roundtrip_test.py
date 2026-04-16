"""End-to-end test: save a tiny real checkpoint, load it, play one move over UCI."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import chess
import pytest
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from halluci_mate.chess_tokenizer import ChessTokenizer
from halluci_mate.inference import ChessInferenceEngine
from halluci_mate.uci_engine import UciEngine

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """Build and save a 2-layer Qwen3 model sized to the chess vocabulary."""
    tokenizer = ChessTokenizer()
    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        head_dim=16,
    )
    model = Qwen3ForCausalLM(config)
    checkpoint = tmp_path / "tiny"
    model.save_pretrained(str(checkpoint))
    return checkpoint


def test_from_checkpoint_loads_and_emits_legal_bestmove(tiny_checkpoint: Path) -> None:
    """Load a saved checkpoint via the inference engine, run one UCI go, verify the bestmove is legal."""
    engine = ChessInferenceEngine.from_checkpoint(tiny_checkpoint, device="cpu")

    stdin = io.StringIO("uci\nisready\nposition startpos\ngo\nquit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    UciEngine(engine=engine, stdin=stdin, stdout=stdout, stderr=stderr).run()

    output = stdout.getvalue()
    assert "uciok" in output
    assert "readyok" in output
    bestmove_line = next(line for line in output.splitlines() if line.startswith("bestmove "))
    move_uci = bestmove_line.split()[1]
    assert chess.Move.from_uci(move_uci) in chess.Board().legal_moves


def test_auto_model_roundtrip(tiny_checkpoint: Path) -> None:
    """Sanity check: the saved checkpoint loads cleanly via AutoModelForCausalLM."""
    model = AutoModelForCausalLM.from_pretrained(str(tiny_checkpoint))
    assert model.config.vocab_size == ChessTokenizer().vocab_size
