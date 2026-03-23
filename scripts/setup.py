"""Setup script to create a Qwen3-0.6B model resized for ChessTokenizer.

Creates a fresh model (random weights) with the Qwen3-0.6B architecture,
resized to match the ChessTokenizer vocabulary (~1,796 tokens).
"""

from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM

from halluci_mate.chess_tokenizer import ChessTokenizer


def main(
    model_name: str = "Qwen/Qwen3-0.6B",
    output_directory: Path = Path("model"),
) -> None:
    """Setup the base model and tokenizer for chess training.

    Loads the Qwen3-0.6B architecture config, creates a fresh model with
    random weights, and resizes it for the ChessTokenizer vocabulary.

    Args:
        model_name: HuggingFace model ID for architecture config.
        output_directory: Where to save the model and tokenizer.
    """
    tokenizer = ChessTokenizer()

    # Load architecture config only (no pretrained weights)
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = tokenizer.vocab_size

    # Create model with random weights
    model = AutoModelForCausalLM.from_config(config)

    output_directory.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_directory)
    model.save_pretrained(output_directory)

    print(f"Saved model and tokenizer to {output_directory}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")


if __name__ == "__main__":
    main()
