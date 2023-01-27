import os
import pickle
import sys
import tensorflow as tf
from pgn2data import Train_Data_Factor, Max_Score, Board_Repr_Size, Sub_Dirs
from pgn2data import Train_Data_File, Val_Data_File, Train_Label_File, Val_Label_File, Items_Subdir_Count_File


BATCH_SIZE = 512
EPOCHS = 10
Min_Steps_Per_Epoch = int(1 / (1 - Train_Data_Factor)) # So that validation will have at least 1 step
AUTO = tf.data.experimental.AUTOTUNE

def read_data(tf_bytestring):
    board = tf.io.decode_raw(tf_bytestring, tf.uint8)
    board = tf.reshape(board, [Board_Repr_Size])
    return board

def read_label(tf_bytestring):
    score = tf.io.decode_raw(tf_bytestring, tf.int32)
    score = tf.cast(score, tf.float32) / Max_Score # normalize 0.0 .. 1.0
    return score

def load_dataset(data_file, label_file):
    data_dataset = tf.data.FixedLengthRecordDataset(filenames=[data_file],
        record_bytes=Board_Repr_Size, header_bytes=0, footer_bytes=0)
    data_dataset = data_dataset.map(read_data, num_parallel_calls=16)

    label_dataset = tf.data.FixedLengthRecordDataset(filenames=[label_file],
        record_bytes=4, header_bytes=0, footer_bytes=0)
    label_dataset = label_dataset.map(read_label, num_parallel_calls=16)

    dataset = tf.data.Dataset.zip((data_dataset, label_dataset))

    #for data in data_dataset:
    #    print(data)
    #exit()

    return dataset

def make_training_dataset(data_file, label_file):
    dataset = load_dataset(data_file, label_file)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # important on TPU, batch size must be fixed
    dataset = dataset.prefetch(AUTO)  # fetch next batches while training on the current one
    return dataset


def make_validation_dataset(data_file, label_file):
    dataset = load_dataset(data_file, label_file)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.prefetch(AUTO)
    return dataset


def  main(model_dir, data_dir, total_items_count):
    file = open(data_dir + '/' + str(total_items_count) + '/' + Items_Subdir_Count_File, 'rb')
    items_subdir_count = pickle.load(file)
    #print(items_subdir_count)
    file.close()

    for sub_dir in Sub_Dirs:
        steps_per_epoch = int(items_subdir_count[sub_dir] * Train_Data_Factor) // BATCH_SIZE
        print(items_subdir_count[sub_dir] , steps_per_epoch)
        if steps_per_epoch < Min_Steps_Per_Epoch:
            print("Warning: insufficient number of steps_per_epoch=%d for sub_dir=%s", (steps_per_epoch, sub_dir))
            continue

        data_sub_dir = data_dir + '/' + str(total_items_count) + '/' + sub_dir
        model_sub_dir = model_dir + '/' + str(total_items_count) + '/' + sub_dir
        train_model(model_sub_dir, data_sub_dir, steps_per_epoch)


def train_model(model_sub_dir, data_sub_dir, steps_per_epoch):
    print('Processing ', data_sub_dir)
    training_dataset = make_training_dataset(data_sub_dir + '/' + Train_Data_File,
                                             data_sub_dir + '/' + Train_Label_File)
    validation_dataset = make_validation_dataset(data_sub_dir + '/' + Val_Data_File,
                                                 data_sub_dir + '/' + Val_Label_File)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[Board_Repr_Size, 1]),
        #tf.keras.layers.Dense(1000, activation="relu"), # This layer gives only a small (0.003) improvement for 10M
        tf.keras.layers.Dense(400, activation="relu"),
        #tf.keras.layers.Dropout(.1, input_shape=(400,)),
        tf.keras.layers.Dense(200, activation="relu"),
        #tf.keras.layers.Dropout(.1, input_shape=(200,)),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid'), #'linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    #model.summary()

    # Train the model
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )

    #steps_per_epoch = int(items_count * Train_Data_Factor) // BATCH_SIZE
    print("Steps per epoch: ", steps_per_epoch)
    try:
        model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
              validation_data=validation_dataset, validation_steps=1,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received!")
        exit()

    model.load_weights(checkpoint_filepath)
    if model_sub_dir is not None:
        print('Saving model ...')
        os.makedirs(model_sub_dir, exist_ok=True)
        model.save(model_sub_dir)
    else:
        print('Model not saved')


# export TF_CPP_MIN_LOG_LEVEL=2
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 model.py model-dir data-dir items-count")
        exit()

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))




