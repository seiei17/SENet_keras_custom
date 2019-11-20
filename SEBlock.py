# define SE Block
import keras.backend as K
from keras.layers import GlobalAvgPool2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Multiply
from keras.layers import Reshape


def _default_Fsq(input):
    fsq = GlobalAvgPool2D()(input)
    return fsq


def _default_Fex(input, ratio=16):
    shape = K.int_shape(input)
    fc1 = Dense(shape[1] // ratio)(input)
    ac1 = Activation('relu')(fc1)
    fc2 = Dense(shape[1])(ac1)
    ac2 = Activation('sigmoid')(fc2)
    return ac2


def se_block(input, Fsq=None, Fex=None, ratio=16):
    if Fsq == None:
        Fsq = _default_Fsq
    if Fex == None:
        Fex = _default_Fex
    shape = K.int_shape(input)
    fsq = Fsq(input)
    fex = Fex(fsq, ratio)
    fex = Reshape((1, 1, shape[3]))(fex)
    scalar = Multiply()([input, fex])
    return scalar