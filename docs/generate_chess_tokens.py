"""
Script to generate the complete vocabulary of legal UCI chess moves.

This generates tokens covering:
- All geometric moves (Queen + Knight patterns cover everything)
- Special control tokens for game outcomes and metadata
"""
import chess


def generate_chess_tokens():
    moves = set()

    # Generate all "To-From" moves (Slides, Steps, Knight Jumps)
    # We iterate through every square and place a "Super Piece" (Queen + Knight)
    # This covers every possible geometric move in chess (orthogonals, diagonals, L-jumps)
    for src in chess.SQUARES:
        for dst in chess.SQUARES:
            if src == dst:
                continue

            # Check if this move is geometrically valid for ANY standard piece
            # We check Queen (covers R, B, P-pushes, K-steps) and Knight
            # Note: We don't check validity against a real board, just geometry.
            dist_rank = abs(chess.square_rank(src) - chess.square_rank(dst))
            dist_file = abs(chess.square_file(src) - chess.square_file(dst))

            is_diagonal = dist_rank == dist_file
            is_straight = dist_rank == 0 or dist_file == 0
            is_knight = (dist_rank == 2 and dist_file == 1) or (
                dist_rank == 1 and dist_file == 2
            )

            if is_diagonal or is_straight or is_knight:
                move = chess.Move(src, dst)
                moves.add(move.uci())

    # Add Special Outcome/Control Tokens
    special_tokens = [
        "<PAD>",
        "<WIN>",
        "<LOSS>",
        "<DRAW>",
        "<WHITE>",
        "<BLACK>",
        "<SEP>",  # Separator between metadata and moves
    ]

    # Sort for consistency
    sorted_vocab = special_tokens + sorted(list(moves))

    return sorted_vocab


if __name__ == "__main__":
    vocab = generate_chess_tokens()
    print(f"Total Tokens Generated: {len(vocab)}")
    print(f"Sample (first 10): {vocab[:10]}")
    print(f"Sample (moves): {vocab[20:30]}")

    # Optionally save to a file
    with open("chess_vocab.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")
