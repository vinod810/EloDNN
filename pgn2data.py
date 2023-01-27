import os
import pickle
import sys
import chess.pgn

Train_Data_Factor = 0.9
Board_Repr_Size = 64 * 12 #* 2
Min_Elo = 800
Max_Elo = 2400
Elo_Bin_Size = 100
Valid_Time_Control = 300 # 5 min
Max_Incr = 5
Min_Clock = 60 # Moves discarded if the clock is below this value
Max_Score = 600 # http://talkchess.com/forum3/download/file.php?id=869
Sub_Dirs = ['elo' + str(x) for x in range(Min_Elo, Max_Elo + 1, Elo_Bin_Size)]

Train_Data_File = 'train_data.dat'
Val_Data_File = 'val_data.dat'
Train_Label_File = 'train_label.dat'
Val_Label_File = 'val_label.dat'
Items_Subdir_Count_File = 'items_subdir_count.data'

def get_board_representation(board, perspective):
    colors = (chess.WHITE, chess.BLACK)
    pieces = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    if perspective == chess.BLACK:
        pieces.reverse()

    board_repr = [0] * Board_Repr_Size
    base_square = 0

    for color in colors:
        for piece in pieces:
            squares = board.pieces(piece, color)
            # piece positions
            for square in squares:
                board_repr[base_square + square] = 1
                # positions attacked by the piece
                #for target in board.attacks(square):
                    #board_repr[base_square + 64 + target] = 1 # FIXME = to +
                # TODO Add BLACK support
            base_square += 64 #* 2

    if perspective == chess.BLACK:
        board_repr.reverse()

    return board_repr


def print_board_representation(board_repr, perspective):
    print("perspective=", 'White' if perspective == chess.WHITE else 'Black')
    for layer in range(int(Board_Repr_Size / 64)):
        board = board_repr[layer * 64: layer * 64 + 64]
        for rows in reversed(range(0, 8)):
            for columns in range(0, 8):
                print(board[rows*8 + columns], end =" ")
            print('')
        print('---------------------------')


def get_score(node, perspective):
    score = 0
    if node.eval().is_mate():
        if perspective == chess.WHITE:
            if node.eval().white().mate() > 0:
                score = Max_Score
            elif node.eval().white().mate() < 0:
                score = -Max_Score
            else:
                assert "Should not be here"
        else:
            if node.eval().black().mate() > 0:
                score = Max_Score
            elif node.eval().black().mate() < 0:
                score = -Max_Score
            else:
                assert "Should not be here"
    else:
        if perspective == chess.WHITE:
            score = node.eval().white().score()
        else:
            score = node.eval().black().score()

    score = min(score, Max_Score)
    score = max(score, -Max_Score)

    return score


def get_prev_turn(node):
    return chess.WHITE if node.turn() == chess.BLACK else chess.BLACK


def  main(pgn_file, data_dir, target_items):
    items_count = 0
    progress_step = target_items / 100
    target_train_items = int(target_items * Train_Data_Factor)

    items_subdir_count = {}
    for sub_dir in Sub_Dirs:
        os.makedirs(data_dir + '/' + str(target_items)  + '/' + sub_dir, exist_ok=True)
        items_subdir_count[sub_dir] = 0

    file_handles = {} # dict of 4 file handles per ELO range
    try:
        for sub_dir in Sub_Dirs:
            dir_name = data_dir + '/' + str(target_items)  + '/' + sub_dir
            file_handles[sub_dir]  = {'train_data':  open(dir_name + '/' + Train_Data_File, 'wb'),
                                      'val_data':    open(dir_name + '/' + Val_Data_File, 'wb'),
                                      'train_label': open(dir_name + '/' + Train_Label_File, 'wb'),
                                      'val_label':   open(dir_name + '/' + Val_Label_File, 'wb')}
        pgn = open(pgn_file)
        while (game := chess.pgn.read_game(pgn)) is not None and items_count < target_items:

            #print(game.headers)
            # Discard Variant games
            for key in game.headers.keys():
                if "Variant" in key:
                    continue

            black_elo = int(round(int(game.headers['BlackElo']) / 100.0, 0) * 100)
            white_elo = int(round(int(game.headers['WhiteElo']) / 100.0, 0) * 100)
            if (not Min_Elo <= black_elo <= Max_Elo) and (not Min_Elo <= white_elo <= Max_Elo):
                continue

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

            #print(game.headers)
            url = game.headers['Site']
            node = game
            while (node := node.next()) is not None:

                if node.eval() is None: # discard the pgn is no engine analysis available
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
                    if prev_clock < Min_Clock: # both players are low on clock
                        break
                    else: # only the player just moved is only low on clock
                        continue

                if abs(prev_score) >= Max_Score: # Ignore completely winning and losing positions
                    continue

                #if get_prev_turn(node) == chess.BLACK:
                    #continue

                score = get_score(node, get_prev_turn(node)) # from the perspective of the player just moved
                gain = score - prev_score
                if gain > 0: # This is due to an engine error
                    continue

                loss = max(0, -gain)
                loss = min(Max_Score, loss)
                #print(node.ply(), get_prev_turn(node), clock, node.move, prev_score, score, loss)

                # write the 'loss' and 'board' to files
                elo =  white_elo if get_prev_turn(node) == chess.WHITE  else black_elo
                if not (Min_Elo <= elo <= Max_Elo):
                    continue

                #print_board_representation(prev_board_repr, get_prev_turn(node) )
                sub_dir = 'elo' + str(elo)
                byte_array = bytearray(prev_board_repr)

                if items_count < target_train_items:
                    file_handles[sub_dir]['train_label'].write(loss.to_bytes(4, byteorder='little', signed=False))
                    file_handles[sub_dir]['train_data'].write(byte_array)
                else:
                    file_handles[sub_dir]['val_label'].write(loss.to_bytes(4, byteorder='little', signed=False))
                    file_handles[sub_dir]['val_data'].write(byte_array)

                items_count += 1
                items_subdir_count[sub_dir] += 1
                if items_count % progress_step == 0:
                    print(int(items_count / progress_step), end=".", flush=True)

    except  KeyboardInterrupt:
        print("KeyboardInterrupt received")

    # close file handles
    for sub_dir in file_handles.keys():
        for file_type in file_handles[sub_dir].keys():
            file_handles[sub_dir][file_type].close()

    # Save number of items per elo
    file = open(data_dir + '/' + str(target_items) + '/' + Items_Subdir_Count_File, 'wb')
    pickle.dump(items_subdir_count, file)
    file.close()

    print("\nitems_count:", items_count)
    print("items_subdir_count:", items_subdir_count)
    if items_count < target_items:
        print("Warning: insufficient positions")


# zstdcat ~/Downloads/lichess_db_standard_rated_2022-11.pgn.zst | python3 pgn2data.py /dev/stdin data 10000
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 pgn2data.py pgn-file, data-dir, number-positions")
        exit()

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))




