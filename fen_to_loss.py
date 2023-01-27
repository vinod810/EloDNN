import sys
import chess.pgn
import tensorflow as tf

from pgn2data import get_board_representation, Max_Score


def  main(model_dir):
    model = tf.keras.models.load_model(model_dir)
    model.summary()

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break
            board = chess.Board(fen)
            board_repr = get_board_representation(board, board.turn)
            loss = model.predict([board_repr], steps=1)
            loss = round((loss[0][0] * Max_Score) / 100, 1)
            print("predicted loss: ", loss)
        except KeyboardInterrupt:
            print("Keyboard Interrupt, exiting")
            exit()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py model-dir")
        exit()

    main(sys.argv[1])
