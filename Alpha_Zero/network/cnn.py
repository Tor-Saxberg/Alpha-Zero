import os, sys, glob; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.compat.v1 import logging; logging.set_verbosity(logging.ERROR)
from time import time

#import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout
from keras.initializers import TruncatedNormal, RandomUniform
from keras.optimizers import Adam
from keras.metrics import mse, categorical_accuracy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model

from PIL import Image

class CNN():
    def __init__(self, env):
        self.game = env.name
        self.board_size = env.board_size
        self.action_size = env.action_size

        inputs = Input(shape=(self.board_size[0], self.board_size[1]*3) ) # (6,7*3)
        net = Reshape((self.board_size[0], self.board_size[1]*3, 1) )(inputs) # (6,21,1)
        for _ in range(1):
            net = Conv2D(128, kernel_size=4, strides=1, padding='same',
                    kernel_initializer=RandomUniform() )(net)
            net = BatchNormalization(axis=3)(net)
            net = Activation('relu')(net)
        net = Flatten()(net)

        for _ in range(2):
            #net = Dense(int(1024/(4**i)),
            net = Dense(64,
                    kernel_initializer=RandomUniform()  )(net)
            net = BatchNormalization(axis=1)(net)
            net = Activation('relu')(net)
            net = Dropout(rate=0.3)(net)
        Q = Dense(1, activation='sigmoid',
                kernel_initializer=RandomUniform(), name='Q')(net)
        pi = Dense(self.action_size, activation='softmax',
                kernel_initializer=RandomUniform(), name='pi')(net)

        self.model = Model(inputs=inputs, outputs=[Q, pi])
        self.model.compile(loss=['mean_squared_error','categorical_crossentropy'],
                           optimizer=Adam(0.001))

        """
        print(self.model.summary())
        plot_model(self.model, to_file='model_plot.png',
                show_shapes=True, show_layer_names=True)
        img = Image.open('model_plot.png')
        img.show()
        """

    def train(self, examples, virtual, epoch=0):
        """train network on batch of examples"""
        boards_list = []; Qs_list = []; policies_list = []
        for example in examples:
            boards, Qs, policies = example
            boards_list.extend(boards)
            Qs_list.extend(Qs)
            policies_list.extend(policies)

        boards = np.reshape(boards_list, (-1,*boards[0].shape) )
        Qs = np.reshape(Qs_list, (-1,1))
        policies = np.reshape(policies_list, (-1,self.action_size))

        self.model.fit(x = boards, y = [Qs, policies],
                       validation_split=0.15,
                       batch_size=boards.size,
                       epochs = 50, shuffle=False, verbose=0,
                       callbacks=[checkpointer, tensorboard])

    def save_checkpoint(self, epoch=0):
            folder='./checkpoints'
            filename=f'{self.game}_cnn_{epoch}.hdf5'
            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                print(f"Making Directory {folder}")
                os.mkdir(folder)
            print(f'saving to {filepath}')
            #self.model.save_weights(filepath)
            return filepath

    def load_checkpoint(self, epoch=0):
        """load network data."""
        folder='./checkpoints'
        filename=f'{self.game}_cnn_{epoch}.hdf5'
        if epoch=='last': # for testing agent
            list_of_files = glob.glob('./checkpoints/*')
            latest_file = max(list_of_files, key=os.path.getctime)
            #if not latest_file:
            #    print(f"No files in ../checkpointes/*")
            #    return
            print(f"Loading from {latest_file}")
            self.model.load_weights(latest_file)
            return

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No model in path {filepath}")
            return
        print(f'loading from {filepath}')
        self.model.load_weights(filepath)
