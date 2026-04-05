# Metrics Collection Plan

This document outlines the metrics strategy for evaluating halluci-mate model performance across all training phases.

## Overview

The model progresses through three training phases, each with different primary metrics:

| Phase | Training Objective | Primary Metric |
|-------|-------------------|----------------|
| 1. Cross-Entropy Pre-training | Learn move patterns | Validation loss |
| 2. DPO for Legality | Prefer legal moves | Illegal move rate |
| 3. DPO vs Stockfish | Prefer strong moves | Stockfish agreement |

## Metrics by Category

### Core Training Metrics (Phase 1)

Built-in or easy to compute during training.

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Validation loss** | Cross-entropy on held-out sequences | HF Trainer (already tracked) |
| **Perplexity** | exp(loss) - interpretable scale | Compute from loss |
| **Top-K accuracy** | % correct move in top K predictions (K=1,5,10) | Custom eval callback |
| **Learning rate** | Training stability indicator | HF Trainer (already tracked) |

### Chess-Specific Metrics (Phase 2)

Require board state reconstruction via `python-chess`.

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Legal move rate** | % of top-1 predictions that are legal in position | Reconstruct board, check legality |
| **Illegal move rate** | Complement of legal move rate | Primary metric for Phase 2 DPO |
| **Move type distribution** | Breakdown: captures, checks, castles, pawn moves | Categorize predictions |

### Stockfish Metrics (Phase 3)

Require Stockfish binary for position evaluation.

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Stockfish agreement** | % model top move matches SF best move | Run SF on eval positions |
| **Centipawn loss (CPL)** | Avg eval gap between model move and best move | SF eval comparison |
| **Blunder rate** | % moves losing >100 centipawns | Track severe mistakes |

### Win Rate Metrics (Phase 3+)

Require game engine for self-play.

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **Win rate vs baseline** | % games won against previous checkpoint | Self-play arena |
| **Elo estimate** | Absolute strength rating | Matches vs rated engines |
| **Draw rate** | Game outcome distribution | Track game results |

## Stratification

All metrics should be computed with stratification by:

- **ELO bucket**: <1200, 1200-1600, 1600-2000, 2000+
- **Game phase**: opening (moves 1-10), middlegame (11-30), endgame (30+)
- **Opening family**: e4, d4, c4, Nf3, other
- **Result type**: decisive games vs draws

This ensures the model performs consistently across different skill levels and game contexts.

## Implementation Plan

### Step 1: Evaluation Callback

Add top-K accuracy callback to `train.py` for lightweight metrics during training.

### Step 2: Evaluation Module

Create `src/halluci_mate/evaluation.py` with:

```python
def compute_top_k_accuracy(model, dataset, k: list[int] = [1, 5, 10]) -> dict[int, float]:
    """Compute top-K move prediction accuracy."""
    ...

def compute_legal_move_rate(model, dataset) -> float:
    """Compute percentage of legal top-1 predictions."""
    ...

def compute_metrics_stratified(model, dataset, stratify_by: list[str]) -> dict:
    """Compute metrics with stratification."""
    ...
```

### Step 3: Evaluation Script

Create `scripts/evaluate.py` entry point for post-training evaluation:

```bash
uv run python scripts/evaluate.py --checkpoint ./checkpoints/best --metrics all
```

### Step 4: Stockfish Integration

Add Stockfish evaluation for Phase 3 metrics:

```python
def compute_stockfish_agreement(model, dataset, stockfish_path: str) -> float:
    """Compute agreement rate with Stockfish best move."""
    ...

def compute_centipawn_loss(model, dataset, stockfish_path: str) -> float:
    """Compute average centipawn loss vs Stockfish."""
    ...
```

### Step 5: Arena/Self-Play

Implement game-playing infrastructure for win rate metrics.

## Dependencies

```bash
uv add python-chess  # Board state reconstruction, legality checking
```

Stockfish binary must be installed separately for SF metrics.

## Compute Cost Considerations

| Metric Category | Cost | When to Run |
|-----------------|------|-------------|
| Core training metrics | Low | Every N steps |
| Top-K accuracy | Low | End of epoch |
| Legal move rate | Medium | End of epoch |
| Stockfish metrics | High | Post-training only |
| Win rate / arena | Very high | Post-training only |

## Phase Transition Gates

- **Phase 1 → Phase 2**: Validation loss converged, legal move rate measured as baseline
- **Phase 2 → Phase 3**: Illegal move rate < 5% (model rarely predicts illegal moves)
- **Phase 3 complete**: Stockfish agreement > 50% or win rate vs Phase 2 checkpoint > 60%
