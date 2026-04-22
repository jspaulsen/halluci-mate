import random

import chess

from halluci_mate.game import Game, Perspective
from halluci_mate.inference import ChessInferenceEngine

CHECKPOINT = "jspaulsen/halluci-mate-v1a"
SEED = 4042


class TestChessInterface:
    def test_chess_inference(self):
        rng = random.Random(SEED)
        engine = ChessInferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
        move_stack = []

        game = Game(
            board=chess.Board(),
            perspective=Perspective.WHITE,
        )

        first_move = engine.predict(game)
        game.push_move(first_move)
        move_stack.append(first_move)

        # Make a random legal move for black to keep the game going
        legal_moves = list(game.board.legal_moves)
        random_move = rng.choice(legal_moves)
        game.push_move(random_move)
        move_stack.append(random_move)

        # Make another prediction for white after the opponent's move
        second_move = engine.predict(game)
        game.push_move(second_move)
        move_stack.append(second_move)

        # Make _yet another_ random legal move for black
        legal_moves = list(game.board.legal_moves)
        random_move = rng.choice(legal_moves)
        game.push_move(random_move)
        move_stack.append(random_move)

        # Finally, one last prediction for white
        third_move = engine.predict(game)
        game.push_move(third_move)
        move_stack.append(third_move)

        # assert the state of the game is as expected (e.g. the moves were applied correctly)
        assert game.board.move_stack == move_stack
