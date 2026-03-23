"""Unit tests for the ChessTokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from transformers import DataCollatorForLanguageModeling

from halluci_mate.chess_tokenizer import (
    BLACK_TOKEN,
    BLACK_TOKEN_ID,
    DRAW_TOKEN,
    DRAW_TOKEN_ID,
    EOS_TOKEN,
    EOS_TOKEN_ID,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    UNK_TOKEN,
    UNK_TOKEN_ID,
    WHITE_TOKEN,
    WHITE_TOKEN_ID,
    ChessTokenizer,
)

# Expected vocabulary size bounds
# 6 special tokens + ~1792 geometric moves (Queen + Knight patterns)
MIN_VOCAB_SIZE = 1700
MAX_VOCAB_SIZE = 1900


class TestVocabulary:
    """Tests for vocabulary generation and structure."""

    def test_vocab_size_in_expected_range(self) -> None:
        tokenizer = ChessTokenizer()
        assert MIN_VOCAB_SIZE <= tokenizer.vocab_size <= MAX_VOCAB_SIZE

    def test_special_tokens_at_correct_indices(self) -> None:
        tokenizer = ChessTokenizer()
        vocab = tokenizer.get_vocab()

        assert vocab[PAD_TOKEN] == PAD_TOKEN_ID
        assert vocab[UNK_TOKEN] == UNK_TOKEN_ID
        assert vocab[EOS_TOKEN] == EOS_TOKEN_ID
        assert vocab[WHITE_TOKEN] == WHITE_TOKEN_ID
        assert vocab[BLACK_TOKEN] == BLACK_TOKEN_ID
        assert vocab[DRAW_TOKEN] == DRAW_TOKEN_ID

    def test_common_opening_moves_in_vocab(self) -> None:
        """Common opening moves should be in the vocabulary."""
        tokenizer = ChessTokenizer()
        vocab = tokenizer.get_vocab()

        common_moves = ["e2e4", "e7e5", "d2d4", "d7d5", "g1f3", "b8c6", "f1c4", "c8f5"]
        for move in common_moves:
            assert move in vocab, f"Expected {move} in vocabulary"

    def test_knight_moves_in_vocab(self) -> None:
        """Knight L-shaped moves should be in vocabulary."""
        tokenizer = ChessTokenizer()
        vocab = tokenizer.get_vocab()

        knight_moves = ["g1f3", "g1h3", "b1c3", "b1a3", "e4f6", "e4d6", "e4g5", "e4g3"]
        for move in knight_moves:
            assert move in vocab, f"Expected knight move {move} in vocabulary"

    def test_invalid_move_not_in_vocab(self) -> None:
        """Geometrically invalid moves should not be in vocabulary."""
        tokenizer = ChessTokenizer()
        vocab = tokenizer.get_vocab()

        # Same square is not a valid move
        assert "e4e4" not in vocab
        # Non-standard geometry (not diagonal, straight, or knight L-shape)
        assert "a1c4" not in vocab  # 3 ranks, 2 files - neither diagonal nor knight
        assert "a1d3" not in vocab  # 2 ranks, 3 files - neither diagonal nor knight


class TestEncoding:
    """Tests for encoding text to token IDs."""

    def test_encode_single_move(self) -> None:
        tokenizer = ChessTokenizer()
        ids = tokenizer.encode("e2e4", add_special_tokens=False)
        assert len(ids) == 1
        assert ids[0] == tokenizer.get_vocab()["e2e4"]

    def test_encode_move_sequence(self) -> None:
        tokenizer = ChessTokenizer()
        ids = tokenizer.encode("e2e4 e7e5 g1f3", add_special_tokens=False)
        assert len(ids) == 3

    def test_encode_with_special_tokens(self) -> None:
        tokenizer = ChessTokenizer()
        text = f"{WHITE_TOKEN} e2e4 e7e5 {EOS_TOKEN}"
        ids = tokenizer.encode(text, add_special_tokens=False)

        assert len(ids) == 4
        assert ids[0] == WHITE_TOKEN_ID
        assert ids[-1] == EOS_TOKEN_ID

    def test_encode_diagonal_move(self) -> None:
        tokenizer = ChessTokenizer()
        ids = tokenizer.encode("f1c4", add_special_tokens=False)
        assert len(ids) == 1
        assert ids[0] > DRAW_TOKEN_ID  # Should be a move token, not special

    def test_unknown_token_maps_to_unk(self) -> None:
        """Unknown tokens should map to UNK token ID."""
        tokenizer = ChessTokenizer()
        ids = tokenizer.encode("invalid_move", add_special_tokens=False)
        assert ids == [UNK_TOKEN_ID]


class TestDecoding:
    """Tests for decoding token IDs back to text."""

    def test_decode_single_move(self) -> None:
        tokenizer = ChessTokenizer()
        vocab = tokenizer.get_vocab()
        decoded = tokenizer.decode([vocab["e2e4"]])
        assert decoded == "e2e4"

    def test_decode_move_sequence(self) -> None:
        tokenizer = ChessTokenizer()
        vocab = tokenizer.get_vocab()
        ids = [vocab["e2e4"], vocab["e7e5"], vocab["g1f3"]]
        decoded = tokenizer.decode(ids)
        assert decoded == "e2e4 e7e5 g1f3"

    def test_roundtrip_encode_decode(self) -> None:
        """Encoding then decoding should return original text."""
        tokenizer = ChessTokenizer()
        original = "e2e4 e7e5 g1f3 b8c6"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == original

    def test_roundtrip_with_special_tokens(self) -> None:
        tokenizer = ChessTokenizer()
        original = f"{WHITE_TOKEN} e2e4 e7e5 {EOS_TOKEN}"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == original

    def test_roundtrip_drawn_game(self) -> None:
        """Drawn games end with DRAW token before EOS."""
        tokenizer = ChessTokenizer()
        original = f"{WHITE_TOKEN} e2e4 e7e5 {DRAW_TOKEN} {EOS_TOKEN}"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == original
        assert ids[-2] == DRAW_TOKEN_ID
        assert ids[-1] == EOS_TOKEN_ID

    def test_roundtrip_edge_rank_moves(self) -> None:
        tokenizer = ChessTokenizer()
        original = "e7e8 a2a1"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == original


class TestHuggingFaceCompat:
    """Tests for HuggingFace Transformers compatibility."""

    def test_pad_token_id_attribute(self) -> None:
        tokenizer = ChessTokenizer()
        assert tokenizer.pad_token_id == PAD_TOKEN_ID

    def test_eos_token_id_attribute(self) -> None:
        tokenizer = ChessTokenizer()
        assert tokenizer.eos_token_id == EOS_TOKEN_ID

    def test_unk_token_id_attribute(self) -> None:
        tokenizer = ChessTokenizer()
        assert tokenizer.unk_token_id == UNK_TOKEN_ID

    def test_vocab_size_property(self) -> None:
        tokenizer = ChessTokenizer()
        assert tokenizer.vocab_size == len(tokenizer.get_vocab())

    def test_len_returns_vocab_size(self) -> None:
        tokenizer = ChessTokenizer()
        assert len(tokenizer) == tokenizer.vocab_size

    def test_data_collator_integration(self) -> None:
        """DataCollatorForLanguageModeling should work with this tokenizer."""
        tokenizer = ChessTokenizer()
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Create sample batch
        sample1 = tokenizer("e2e4 e7e5", return_tensors="pt", padding=True)
        sample2 = tokenizer("d2d4 d7d5 c2c4", return_tensors="pt", padding=True)

        batch = collator([{"input_ids": sample1["input_ids"][0]}, {"input_ids": sample2["input_ids"][0]}])

        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[0] == 2  # Batch size

    def test_save_vocabulary(self) -> None:
        """Vocabulary should be saveable to JSON."""
        tokenizer = ChessTokenizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = tokenizer.save_vocabulary(tmpdir)
            assert len(saved_files) == 1

            vocab_path = Path(saved_files[0])
            assert vocab_path.exists()
            assert vocab_path.suffix == ".json"

    def test_model_input_names(self) -> None:
        """Should have standard model input names."""
        tokenizer = ChessTokenizer()
        assert "input_ids" in tokenizer.model_input_names
        assert "attention_mask" in tokenizer.model_input_names
