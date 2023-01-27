import sys
import chess.pgn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pgn2data import get_board_representation, Max_Score, Sub_Dirs, Board_Repr_Size
from model import read_data

model_dict = {}

def get_move_elo(model_dir, items_count, board_repr, actual_loss):
    #tf.compat.v1.disable_eager_execution()

    x = []
    y = []
    for sub_dir in Sub_Dirs:
        #path = model_dir + '/' + str(items_count) + '/' + sub_dir
        predicted_loss = estimate_loss(board_repr, sub_dir)
        x.append(predicted_loss)
        elo = int(sub_dir[3:])
        y.append(elo)

    #print(x)
    #print(y)
    #plt.plot(x, y)
    #plt.show()
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return max(0, int(round(reg.predict([[actual_loss]])[0], 0)))


def estimate_loss(board_repr, sub_dir):
    model = model_dict[sub_dir]  # tf.keras.models.load_model(path)
    loss = model.predict([board_repr], steps=1, verbose=0)
    loss = int(round(loss[0][0] * Max_Score, 0))
    return loss


def  main(model_dir, items_count):

    for sub_dir in Sub_Dirs:
        path = model_dir + '/' + str(items_count) + '/' + sub_dir
        model = tf.keras.models.load_model(path)
        model_dict[sub_dir] = model

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break
            loss = input("Loss CP: ")
            loss = int(loss)
        except KeyboardInterrupt:
            print("Keyboard Interrupt, exiting")
            exit()

        board = chess.Board(fen)
        board_repr = get_board_representation(board, board.turn)
        #data_dataset = tf.data.Dataset.from_tensor_slices(board_repr)
        #data_dataset = data_dataset.map(read_data, num_parallel_calls=1)

        move_elo = get_move_elo(model_dir, items_count, board_repr, loss)
        print(move_elo)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 fen_to_elo model-dir items-count")
        exit()

    main(sys.argv[1], int(sys.argv[2]))
