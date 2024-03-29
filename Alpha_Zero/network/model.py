import keras.backened as K
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.regularizers import l2

#from connect4_zero.config import Config

class ModelConfig:
    cnn_filter_num = 64
    cnn_filter_size = 4
    res_layer_num = 3
    l2_reg = 1e-4
    value_fc_size = 32

class Connect4Model:
    def __init__(self)
        self.config = ModelConfig()
        self.model = None  # type: Model
        self.digest = None

    def build(self):
        mc = self.config
        in_x = x = Input((2,6,7))  # [own(8x8), enemy(8x8)] if Input((2,6,7))
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size,
                padding="same", data_format="channels_first",
                kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first",
                kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(mc.n_labels, kernel_regularizer=l2(mc.l2_reg),
                activation="softmax", name="policy_out")(x)

         # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first",
                kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg),
                activation="relu")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
                activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="connect4_model")

    def _build_residual_block(self, x):
        mc = self.config
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num,
                kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num,
                kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x
