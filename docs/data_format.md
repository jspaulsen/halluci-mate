# Data Format & Sequence Transformation

This document describes how raw Lichess games are transformed into training sequences.

## Sequence Format

Each training example is a sequence of UCI moves from one player's perspective:

```
<COLOR> move1 move2 move3 ... [<DRAW>] <EOS>
```

- `<COLOR>`: `<WHITE>` or `<BLACK>` - the player whose perspective this sequence represents
- Moves are interleaved, but ordered so the perspective player's moves come first in each pair
- `<DRAW>`: Optional, present only for drawn games
- `<EOS>`: End of sequence

### Won Games

For decisive games (win/loss), generate **one** training example from the winner's perspective.

**White wins:**
```
<WHITE> e2e4 e7e5 g1f3 b8c6 ... <EOS>
```
Moves in standard order (white first).

**Black wins:**
```
<BLACK> e7e5 e2e4 b8c6 g1f3 ... <EOS>
```
Moves reordered so black's moves come first in each pair.

### Drawn Games

For draws, generate **two** training examples - one from each perspective.

**Same game, two sequences:**
```
<WHITE> e2e4 d7d5 c2c4 ... <DRAW> <EOS>
<BLACK> d7d5 e2e4 c7c6 c2c4 ... <DRAW> <EOS>
```

Rationale: In draws, neither player "lost" - both played well enough to not lose. Training from both perspectives ensures the model learns solid play from either side.

## Transformation Pseudocode

```python
def transform_game(moves: list[str], outcome: str) -> list[Sequence]:
    """
    Transform a raw game into training sequence(s).

    Args:
        moves: UCI moves in standard order (white first): ["e2e4", "e7e5", ...]
        outcome: "white", "black", or "draw"

    Returns:
        List of training sequences (1 for decisive, 2 for draws)
    """
    sequences = []

    if outcome == "white":
        # White won: single sequence from white's perspective
        sequences.append(format_sequence(moves, perspective="white"))

    elif outcome == "black":
        # Black won: single sequence from black's perspective
        sequences.append(format_sequence(moves, perspective="black"))

    elif outcome == "draw":
        # Draw: both perspectives
        sequences.append(format_sequence(moves, perspective="white", is_draw=True))
        sequences.append(format_sequence(moves, perspective="black", is_draw=True))

    return sequences


def format_sequence(moves: list[str], perspective: str, is_draw: bool = False) -> str:
    """
    Format moves into a training sequence.

    Args:
        moves: UCI moves in standard order
        perspective: "white" or "black"
        is_draw: Whether to append <DRAW> token
    """
    if perspective == "white":
        # Standard order: white's moves at even indices (0, 2, 4, ...)
        ordered_moves = moves
        color_token = "<WHITE>"
    else:
        # Reorder: black's moves first in each pair
        white_moves = moves[0::2]  # indices 0, 2, 4, ...
        black_moves = moves[1::2]  # indices 1, 3, 5, ...
        ordered_moves = interleave(black_moves, white_moves)
        color_token = "<BLACK>"

    tokens = [color_token] + ordered_moves
    if is_draw:
        tokens.append("<DRAW>")
    tokens.append("<EOS>")

    return " ".join(tokens)
```

## Why This Format?

**Leading color token = "play as this player"**

The model learns to generate moves that lead to favorable outcomes for the indicated color. For wins, that color won. For draws, that color held the draw.

**Winner's perspective only for decisive games**

Training on the loser's perspective would teach the model to play losing chess. We only want it to learn from winning/drawing play.

**Both perspectives for draws**

Draws are symmetric - neither side made a losing mistake. The model should learn solid, drawing play from both colors.

## Token Vocabulary

| Token | Index | Meaning |
|-------|-------|---------|
| `<PAD>` | 0 | Padding |
| `<UNK>` | 1 | Unknown token |
| `<EOS>` | 2 | End of sequence |
| `<WHITE>` | 3 | White's perspective |
| `<BLACK>` | 4 | Black's perspective |
| `<DRAW>` | 5 | Game ended in draw |
| `a1a2`, `a1a3`, ... | 6+ | UCI moves (~1792 tokens) |

Total vocabulary: ~1798 tokens.
