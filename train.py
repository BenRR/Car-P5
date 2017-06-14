import os
import csv
import cv2
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Lambda
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

# data location and training set
# training set assumed in csv file with 2 columns
# img_file_with_path,vehicle or not(1,0)
# use data_augment.py to generate training set
# by default use current directory and assume data file is called training_vehicle.csv and training_non_vehicle.csv
data_dir = os.getcwd()

# use the final folder generated from data_augment
final_data_folder = '{}/training'.format(data_dir)
training_positive_file = '{}/vehicle.csv'.format(final_data_folder)
training_negative_file = '{}/non_vehicle.csv'.format(final_data_folder)
img_folder = final_data_folder

# model files and if load previous model
model_file = 'car_model.h5'
previous_model = 'car_previous.h5'

# hyper-parameters
epoch_num = 35
default_batch_size = 269
default_validation_size = 0.3


# generator
def generator(training_samples, batch_size=default_batch_size):
    num_samples = len(training_samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(training_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = training_samples[offset:offset+batch_size]

            images = []
            cars = []
            for batch_sample in batch_samples:
                img_file = batch_sample[0]
                image = cv2.imread(img_file)
                car = int(batch_sample[1])
                # print(img_file)
                assert car == 1 or car == 0
                assert image.shape == (64, 64, 3)
                images.append(image)
                cars.append(car)

            X_train = np.array(images)
            y_train = np.array(cars)
            yield sklearn.utils.shuffle(X_train, y_train)


# load training data
def load_training(file):
    training_samples = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            training_samples.append(line)
    return training_samples


# model
def create_model(pre_load_weights = False, input_shape =(64, 64, 3), weights = previous_model):
    # dropout rate
    default_drop_out_rate = 0.5

    #  lenet
    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape = input_shape))
    model.add(BatchNormalization())

    # Conv layer with elu activation and max pooling
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation="elu"))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(BatchNormalization())
    model.add(Dropout(default_drop_out_rate))

    # Conv layer with elu activation and max pooling
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation="elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # Full conv with elu activation
    model.add(Dropout(default_drop_out_rate))
    model.add(Convolution2D(1024, 8, 8, activation="elu"))
    model.add(BatchNormalization())

    # Full conv with elu activation
    model.add(Dropout(default_drop_out_rate / 2))
    model.add(Convolution2D(100, 1, 1, activation="elu"))
    model.add(BatchNormalization())

    # Full conv with elu activation
    model.add(Convolution2D(1, 1, 1, activation="sigmoid"))

    # load other previous model if needed
    if pre_load_weights:
        print("load previous weights!")
        model.load_weights(weights)

    return model


def train(tr_generator, vl_generator, tr_size, vl_size):
    model = create_model(pre_load_weights = False)
    model.add(Flatten())

    # checkpoints. save all models that improve the validation loss
    filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    tb = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, tb]

    # use absolute error and adam optimizer
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    # train model
    model.fit_generator(tr_generator,
                        samples_per_epoch=tr_size,
                        validation_data=vl_generator,
                        nb_val_samples=vl_size,
                        callbacks=callbacks_list,
                        nb_epoch=epoch_num, verbose=1)

    # save final model
    model.save(model_file)


def prepare_data():
    positive_samples = load_training(training_positive_file)
    negative_samples = load_training(training_negative_file)

    np.random.shuffle(positive_samples)
    np.random.shuffle(negative_samples)
    positive_train_samples, positive_validation_samples = train_test_split(positive_samples,
                                                                           test_size=default_validation_size,
                                                                           random_state=42)

    negative_train_samples, negative_validation_samples = train_test_split(negative_samples,
                                                                           test_size=default_validation_size,
                                                                           random_state=42)

    np.random.shuffle(positive_train_samples)
    np.random.shuffle(positive_validation_samples)
    np.random.shuffle(negative_train_samples)
    np.random.shuffle(negative_validation_samples)

    all_train_samples = np.append(negative_train_samples, positive_train_samples, axis=0)
    all_validation_samples = np.append(positive_validation_samples, negative_validation_samples, axis=0)

    np.random.shuffle(all_train_samples)
    np.random.shuffle(all_validation_samples)

    return all_train_samples, all_validation_samples


if __name__ == '__main__':

    train_samples, validation_samples = prepare_data()

    print(len(train_samples))
    print(len(validation_samples))

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=default_batch_size)
    validation_generator = generator(validation_samples, batch_size=default_batch_size)

    train(train_generator, validation_generator, len(train_samples), len(validation_samples))
