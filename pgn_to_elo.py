import sys
import chess.pgn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pgn2data import get_board_representation, get_score, get_prev_turn
from pgn2data import Max_Score, Sub_Dirs, Valid_Time_Control, Max_Incr, Min_Clock

Min_Moves_Elo = 20
model_dict = {}

def get_move_elo(board_repr, actual_loss):
    x = []
    y = []
    for sub_dir in Sub_Dirs:
        predicted_loss = estimate_loss(board_repr, sub_dir)
        x.append(predicted_loss)
        elo = int(sub_dir[3:])
        y.append(elo)

    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return int(round(reg.predict([[actual_loss]])[0], 0)) #max(0, int(round(reg.predict([[actual_loss]])[0], 0)))


def estimate_loss(board_repr, sub_dir):
    model = model_dict[sub_dir]  # tf.keras.models.load_model(path)
    loss = model.predict([board_repr], steps=1, verbose=0)
    loss = int(round(loss[0][0] * Max_Score, 0))
    return loss


def  main(model_dir, items_count, pgn_file, target_estimates):

    for sub_dir in Sub_Dirs:
        path = model_dir + '/' + str(items_count) + '/' + sub_dir
        model = tf.keras.models.load_model(path)
        model_dict[sub_dir] = model

    elo_X = []
    elo_y = []
    estimate_count = 0
    pgn = open(pgn_file)
    while (game := chess.pgn.read_game(pgn)) is not None and estimate_count < target_estimates:

        for key in game.headers.keys():
            if "Variant" in key:
                continue

        black_actual_elo = int(game.headers['BlackElo'])
        white_actual_elo = int(game.headers['WhiteElo'])

        try:
            time_control = int(game.headers['TimeControl'].split('+')[0])
            time_incr = int(game.headers['TimeControl'].split('+')[1])
        except ValueError:
            continue
        if time_control != Valid_Time_Control or time_incr > Max_Incr:
            continue

        saved_prev_score = 0
        saved_prev_board_repr = get_board_representation(game.board(), chess.WHITE)
        clock = time_control

        url = game.headers['Site']
        node = game
        white_elo_estimates = []
        black_elo_estimates = []
        while (node := node.next()) is not None:

            if node.eval() is None:  # discard the pgn as no engine analysis available
                break

            prev_score = saved_prev_score
            prev_board_repr = saved_prev_board_repr

            # Save values for the next iteration from the perspective of the player about to move
            saved_prev_score = get_score(node, node.turn())
            saved_prev_board_repr = get_board_representation(node.board(), node.turn())

            prev_clock = clock
            clock = node.clock()

            # Consider the move only if the time left on the clock is not too low
            if clock < Min_Clock:
                if prev_clock < Min_Clock:  # both players are low on clock
                    break
                else:  # only the player just moved is only low on clock
                    continue

            if abs(prev_score) >= Max_Score:  # Ignore completely winning and losing positions
                continue

            score = get_score(node, get_prev_turn(node))  # from the perspective of the player just moved
            gain = score - prev_score
            if gain > 0:  # This is due to an engine error
                continue

            loss = max(0, -gain)
            loss = min(Max_Score, loss)

            move_elo = get_move_elo(prev_board_repr, loss)
            #print(url, node.ply(), get_prev_turn(node), clock, node.move, prev_score, score, loss, move_elo)
            if get_prev_turn(node) == chess.WHITE:
                white_elo_estimates.append(move_elo)
            else:
                black_elo_estimates.append(move_elo)

        if len(white_elo_estimates) >= Min_Moves_Elo:
            estimate_count += 1
            white_estimated_elo = int(round(sum(white_elo_estimates) / len(white_elo_estimates), 0))
            print(white_actual_elo, white_estimated_elo)
            elo_X.append(white_estimated_elo)
            elo_y.append(white_actual_elo)

        if len(black_elo_estimates) >= Min_Moves_Elo:
            estimate_count += 1
            black_estimated_elo = int(round(sum(black_elo_estimates) / len(black_elo_estimates), 0))
            print(black_actual_elo, black_estimated_elo)
            elo_X.append(black_estimated_elo)
            elo_y.append(black_actual_elo)

    plt.scatter(elo_X, elo_y)
    plt.show()

    elo_X = np.array(elo_X).reshape((-1, 1))
    elo_y = np.array(elo_y)
    #reg = LinearRegression().fit(elo_X, elo_y)
    r2 = r2_score(elo_X, elo_y, multioutput = 'variance_weighted')
    print("r2=", r2)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python3 pgn_to_elo.py  model-dir items-count, pgn-file target-estimates")
        exit()
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
