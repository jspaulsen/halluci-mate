# Training Strategy

This document outlines the multi-phase training approach for halluci-mate.

## Overview

Training proceeds in three phases, each building on the previous:

1. **Phase 1 - Syntax & Patterns**: Standard cross-entropy on Lichess games
2. **Phase 2 - Legality**: DPO to penalize illegal moves
3. **Phase 3 - Quality**: DPO against Stockfish evaluations

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Phase 1      │     │    Phase 2      │     │    Phase 3      │
│                 │     │                 │     │                 │
│  Cross-Entropy  │────▶│  DPO: Legality  │────▶│ DPO: Stockfish  │
│  (Lichess data) │     │  (legal > illegal)   │  (better > worse)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Phase 1: Cross-Entropy Pre-training

**Goal**: Learn chess move syntax, common patterns, and game flow.

**Approach**: Standard causal language modeling with cross-entropy loss on Lichess game sequences.

**Data format** (see [data_format.md](data_format.md) for details):
```
<WHITE> e2e4 e7e5 g1f3 b8c6 ... <EOS>        (white won)
<BLACK> e2e4 e7e5 g1f3 b8c6 ... <EOS>        (black won)
<WHITE> e2e4 d7d5 ... <DRAW> <EOS>           (draw, white's perspective)
<BLACK> e2e4 d7d5 ... <DRAW> <EOS>           (draw, black's perspective)
```

Moves are always in standard order (white first). Drawn games produce two training examples (one per perspective). Decisive games produce one (winner's perspective only).

**What the model learns**:
- Valid UCI move structure (4-character from-to notation)
- Opening theory and common responses
- Typical game progressions
- Turn-taking patterns

**Loss**: Standard cross-entropy, all tokens weighted equally.

**Considerations**:
- This phase treats predicting an illegal move the same as predicting the wrong legal move
- The model may learn some illegal patterns if they appear in data (unlikely with clean Lichess data, but possible with corrupted games)
- Should establish strong baseline before moving to Phase 2

## Phase 2: DPO for Move Legality

**Goal**: Teach the model to prefer legal moves over illegal ones.

**Approach**: Direct Preference Optimization (DPO) where legal moves are preferred over illegal moves given a board position.

### Why DPO over PPO?

| Aspect | PPO | DPO |
|--------|-----|-----|
| Stability | Requires careful tuning | More stable out-of-box |
| Complexity | Needs reward model + policy training | Single training phase |
| Compute | Higher (multiple forward passes) | Lower |
| Reference model | Separate | Implicit in loss |

DPO directly optimizes the policy without explicitly learning a reward model, making it simpler and more stable for this use case.

### Generating Preference Pairs

For each position in training data:

1. Parse the game to get the current board state
2. The actual move played = **chosen** (assumed legal)
3. Sample an illegal move from the vocabulary = **rejected**

```python
# Pseudocode for generating DPO pairs
def generate_legality_pairs(game_moves: list[str]) -> list[DPOPair]:
    board = chess.Board()
    pairs = []

    for move_uci in game_moves:
        legal_moves = set(m.uci() for m in board.legal_moves)
        illegal_moves = ALL_MOVE_TOKENS - legal_moves

        rejected = random.choice(list(illegal_moves))

        pairs.append(DPOPair(
            prompt=board_to_sequence(board),  # moves so far
            chosen=move_uci,
            rejected=rejected,
        ))

        board.push_uci(move_uci)

    return pairs
```

### Considerations

- **Sampling strategy for rejected moves**: Random illegal moves may be too easy (e.g., moving to occupied squares). Consider sampling "hard negatives" — moves that are geometrically plausible but illegal in context (e.g., moving a pinned piece).

- **Ratio of legal:illegal pairs**: May want to balance with some (legal, legal) pairs where both are valid but one was played — though this bleeds into Phase 3 territory.

- **When to stop**: Monitor the illegal move rate during validation. Diminishing returns once the model rarely predicts illegal moves.

## Alternative: Auxiliary Legality Head

Instead of (or in addition to) Phase 2 DPO, we could add a second prediction head that explicitly learns move legality.

### Architecture

```
                    ┌──────────────────┐
                    │   Transformer    │
                    │   (Qwen3-0.6B)   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │    LM Head      │           │  Legality Head  │
    │  (next token)   │           │  (binary: legal │
    │                 │           │   or illegal)   │
    └─────────────────┘           └─────────────────┘
```

### Training Objective

Joint loss combining next-token prediction and legality classification:

```python
loss = ce_loss(lm_logits, next_token) + alpha * bce_loss(legality_logits, is_legal)
```

Where:
- `lm_logits`: standard next-token prediction over vocabulary
- `legality_logits`: binary prediction for each vocab token (legal/illegal given board state)
- `alpha`: weighting hyperparameter (start with 0.1-0.5)

### Generating Legality Labels

For each position, compute the legal move mask:

```python
def get_legality_labels(board: chess.Board, vocab_size: int) -> torch.Tensor:
    """Returns binary tensor: 1 for legal moves, 0 for illegal."""
    labels = torch.zeros(vocab_size)
    legal_moves = {m.uci() for m in board.legal_moves}

    for move_uci, token_id in tokenizer.get_vocab().items():
        if move_uci in legal_moves:
            labels[token_id] = 1.0

    return labels
```

### Pros

- **Explicit legality signal**: The model learns a direct representation of "what's legal here"
- **Regularization**: The legality head encourages representations that encode board state
- **Inference-time filtering**: Can use legality head to mask/weight LM predictions
- **Single training phase**: Can be done during Phase 1, avoiding a separate DPO phase

### Cons

- **Architecture change**: Requires modifying the model, not just the training loop
- **Compute overhead**: Extra forward pass through legality head
- **Label generation**: Need to compute legal moves for every position during training
- **Calibration**: Legality head confidence may not align with LM head probabilities

### When to Consider This

- If Phase 2 DPO proves unstable or slow to converge
- If we want legality awareness from the start (during Phase 1)
- If we plan to use constrained decoding at inference and want calibrated legality scores

### Implementation Sketch

```python
class ChessLMWithLegalityHead(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.legality_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, legality_labels=None, **kwargs):
        hidden = self.transformer(input_ids, **kwargs).last_hidden_state

        lm_logits = self.lm_head(hidden)
        legality_logits = self.legality_head(hidden)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            if legality_labels is not None:
                bce_loss = F.binary_cross_entropy_with_logits(legality_logits, legality_labels)
                loss = ce_loss + self.config.legality_weight * bce_loss
            else:
                loss = ce_loss

        return CausalLMOutput(loss=loss, logits=lm_logits, legality_logits=legality_logits)
```

---

## Phase 3: DPO Against Stockfish

**Goal**: Teach the model to prefer stronger moves over weaker ones.

**Approach**: DPO where Stockfish-preferred moves are chosen over the moves actually played (when Stockfish disagrees).

### Generating Preference Pairs

For each position:

1. Get the move actually played in the game
2. Query Stockfish for the best move(s)
3. If Stockfish disagrees with the played move, create a preference pair

```python
def generate_stockfish_pairs(game_moves: list[str], engine: chess.engine) -> list[DPOPair]:
    board = chess.Board()
    pairs = []

    for move_uci in game_moves:
        # Get Stockfish's preferred move
        result = engine.analyse(board, chess.engine.Limit(depth=20))
        best_move = result["pv"][0].uci()

        if best_move != move_uci:
            pairs.append(DPOPair(
                prompt=board_to_sequence(board),
                chosen=best_move,      # Stockfish's choice
                rejected=move_uci,     # Human's choice
            ))

        board.push_uci(move_uci)

    return pairs
```

### Open Questions & Considerations

#### 1. Should we always prefer Stockfish?

**Argument for**: Stockfish is objectively stronger; we want the strongest possible play.

**Argument against**:
- Human games have instructive value (openings, practical chances, time pressure decisions)
- Stockfish play can be "inhuman" and hard to learn from
- We may want a model that plays human-like strong chess, not engine chess

**Possible middle ground**: Only create preference pairs when Stockfish evaluation differs significantly (e.g., >1.0 pawn difference). Small differences may be stylistic rather than objective errors.

#### 2. Evaluation depth

- Deeper = more accurate but slower data generation
- Shallow (depth 10-12) may be sufficient for obvious blunders
- Consider using NNUE for speed

#### 3. Risk of mode collapse

- If we only train on Stockfish preferences, the model may converge to a narrow set of "engine-approved" moves
- May lose diversity in openings and playing styles
- Consider mixing in some human preference data or using KL regularization

#### 4. Compute cost

- Running Stockfish on every position in millions of games is expensive
- Could sample positions (e.g., every 5th move, or only middlegame positions)
- Could pre-compute and cache Stockfish evaluations

#### 5. Alternative: Self-play + Stockfish adjudication

Instead of per-move Stockfish comparison, could:
1. Have the model play against itself
2. Use Stockfish to adjudicate game outcomes
3. DPO on (winning game moves > losing game moves)

This is more like AlphaZero's approach but with DPO instead of policy gradient.

#### 6. Do we even need Phase 3?

The model trained on Lichess data (Phase 1) already learns from games of varying skill levels. Phase 2 ensures legality. Phase 3 is about pushing beyond human play.

**Skip Phase 3 if**: The goal is "plays reasonable chess" rather than "plays optimal chess."

**Do Phase 3 if**: We want to exceed human-level play from the training data.

## Evaluation Metrics

Track these across all phases:

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Validation loss | Primary | Monitor | Monitor |
| Illegal move rate | Monitor | Primary | Monitor |
| Stockfish agreement | Baseline | Monitor | Primary |
| Win rate vs baseline | — | — | Primary |
| Elo estimate (arena) | Baseline | Compare | Compare |

## Implementation Notes

### Dependencies for Phase 2+

```toml
# Additional dependencies for DPO training
trl = ">=0.8.0"  # HuggingFace TRL for DPO
python-chess = ">=1.10.0"  # Already have this
```

### Stockfish setup for Phase 3

```bash
# Install Stockfish
sudo apt install stockfish
# Or download NNUE version for better speed/accuracy
```

## Future directions

- [Inference-time search over LM top-K](inference_search.md) — depth-2 minimax
  using the LM as policy and a learned/hand-crafted leaf eval. Stacks
  multiplicatively with Phase 3; out of scope for the three training phases
  but tracked as a candidate next step.

## References

- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [TRL Library](https://github.com/huggingface/trl) - HuggingFace's RL library
- [Stockfish](https://stockfishchess.org/) - Chess engine for Phase 3
