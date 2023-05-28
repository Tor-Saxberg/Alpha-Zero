import os, sys; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import time
from tensorflow.compat.v1 import logging; logging.set_verbosity(logging.ERROR)

from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, Add
from keras.initializers import TruncatedNormal, RandomUniform
from keras.optimizers import Adam
from keras.metrics import mse, categorical_accuracy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model
import numpy as np

from PIL import Image

class RNN():
    def __init__(self, env ):
        self.game = env.name
        self.action_size = env.action_size # an int
        self.board_size = env.board_size # a tuple

        inputs = Input(shape=(self.board_size[0], self.board_size[1]*3) )
        inputs_reshape = Reshape((self.board_size[0], self.board_size[1]*3, 1) )(inputs)
        outer = Conv2D(32, kernel_size=2, strides=1, padding='same')(inputs_reshape)
        outer = BatchNormalization(axis=-1)(outer)
        outer = Activation('relu')(outer)

        inner = outer
        for _ in range(3): # number of blocks
            #inner = outer
            for _ in range(2): # layers before merge
                inner = Conv2D(32, kernel_size=2, strides=1, padding='same')(inner)
                inner = BatchNormalization(axis=-1)(inner)
                inner = Activation('relu')(inner)
            #outer = Add()([inner, outer])
            inner = Add()([inner, outer])

        pi = Conv2D(32, kernel_size=2, strides=1, padding='same')(inner)
        pi = BatchNormalization(axis=-1)(pi)
        pi = Activation('relu')(pi)
        pi = Flatten()(pi)
        pi = Dense(self.action_size, activation='softmax', name='pi_layer')(pi)

        Q = Conv2D(1, kernel_size=1, strides=1, padding='same')(inner)
        Q = BatchNormalization(axis=-1)(Q)
        Q = Activation('relu')(Q)
        Q = Flatten()(Q)
        Q = Dense(1, activation='sigmoid', name='Q_layer')(Q) 

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
        """train network and reeturn win_rate"""
        if not virtual:
            """train on path batch"""
            boards_list = []; Qs_list = []; policies_list = []
            for example in examples: 
                boards, Qs, policies = example
                boards_list.extend(boards)
                Qs_list.extend(Qs)
                policies_list.extend(policies)

            boards = np.reshape(boards_list, (-1,*boards[0].shape) )
            Qs = np.reshape(Qs_list, (-1,1))
            policies = np.reshape(policies_list, (-1,self.action_size))
            print(f"fitting {len(boards)} boards")
            start = time()

            checkpointer = ModelCheckpoint(filepath=self.save_checkpoint(epoch),
                                           save_weights_only=True,
                                           save_best_only=True,
                                           #monitor='loss', mode='min',
                                           verbose=0)
            tensorboard = TensorBoard(log_dir=f'./logging/{self.game}_rnn_{epoch}',
                                      histogram_freq=10,
                                      write_images=True,
                                      batch_size=boards.size,
                                      update_freq='batch')

            self.model.fit(x = boards, y = [Qs, policies],
                           validation_split=0.15,
                           batch_size=boards.size,
                           epochs = 50, shuffle=False, verbose=0,
                           callbacks=[checkpointer, tensorboard])

            end = time()
            print(f"training time: {end - start}")
            sys.stdout.flush()
            #self.save_checkpoint(epoch=epoch)

    def save_checkpoint(self, epoch=0): 
            folder='./checkpoints'
            filename=f'{self.game}_rnn_{epoch}.hdf5'
            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                print(f"Making Directory {folder}")
                os.mkdir(folder)
            print(f'saving to {filepath}')
            #self.model.save_weights(filepath)
            return filepath

    def load_checkpoint(self, epoch=0):
        folder='./checkpoints'
        filename=f'{self.game}_rnn_{epoch}.hdf5'
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f"No model in path {filepath}")
            return
        print(f'loading from {filepath}')
        self.model.load_weights(filepath)

