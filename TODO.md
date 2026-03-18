# TODO

## High Priority

### Tokenizer
- [ ] Create custom chess tokenizer
  - See `docs/generate_chess_tokens.py` for vocabulary generation
  - Covers all geometric moves (Queen + Knight patterns)
  - Special tokens: `<PAD>`, `<WIN>`, `<LOSS>`, `<DRAW>`, `<WHITE>`, `<BLACK>`, `<SEP>`

### Model Setup
- [ ] Load Qwen3-0.6B architecture only (training from scratch, not fine-tuning)
  - Use `AutoConfig.from_pretrained()` + `AutoModelForCausalLM.from_config()`
  - Resize token embeddings to match chess tokenizer vocab size

### Data Processing
- [ ] Implement `_process_chess_game()` function in train.py
  - Parse `movetext` field - two formats to handle:
    - With clock annotations: `1. e4 { [%clk 0:03:00] } 1... d5 { [%clk 0:03:00] } ...`
    - Without clock annotations: `1. e4 e6 2. d4 b6 ...`
  - Strip clock annotations `{ [%clk X:XX:XX] }` and move numbers
  - Convert SAN notation to UCI (e.g., `e4` -> `e2e4`, `Nf3` -> `g1f3`)
    - Use python-chess for board state tracking to resolve ambiguous moves
  - Strip result marker at end (`1-0`, `0-1`, `1/2-1/2`)
- [ ] Sample/limit dataset size (full dataset is ~7B rows, way too large)
  - Decide on reasonable subset size for available disk space

## Medium Priority

- [ ] Validate "Normal" termination filter captures intended games
  - Verify filter excludes: time forfeit, abandonment, rules infraction
- [ ] Decide on train/eval split strategy
- [ ] Review hyperparameters for training from scratch
  - Learning rate, batch size appropriate for from-scratch training?
  - Number of epochs needed?

## Low Priority

- [ ] Consider data augmentation (board flipping, color swapping)
- [ ] Add evaluation metrics specific to chess (move accuracy, legal move %, etc.)
