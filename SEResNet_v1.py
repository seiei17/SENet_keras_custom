# define Net
import keras.backend as K
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import GlobalAvgPool2D
from keras.layers import Dense

from keras.models import Model

from keras.regularizers import l2
from keras.initializers import he_normal as initial

from SEBlock import se_block

seed = None
SE_flag = True
filter_num = 16


def _bn_relu():
    def f(input):
        bn = BatchNormalization(axis=3)(input)
        return Activation('relu')(bn)
    return f


def _conv_bn_relu(filter, k, s=1, pad='same', w_decay=1e-4):
    def f(input):
        conv = Conv2D(filter, k,
                      strides=s,
                      padding=pad,
                      kernel_regularizer=l2(w_decay),
                      kernel_initializer=initial(seed),
                      )(input)
        return _bn_relu()(conv)
    return f


def _shortcut(input, residual, w_decay):
    input_shape = K.int_shape(input)
    res_shape = K.int_shape(residual)
    size_equality = int(round(input_shape[1] / res_shape[1]))
    channel_equality = input_shape[3] == res_shape[3]
    x = input
    if size_equality > 1 or not channel_equality:
        x = Conv2D(res_shape[3], (1, 1),
                   strides=(size_equality, size_equality),
                   padding='same',
                   kernel_regularizer=l2(w_decay),
                   kernel_initializer=initial(seed),
                   )(input)
    return Add()([x, residual])


def _resblock(block, filter, stage, w_decay=1e-4, first_stage=False):
    def f(input):
        for i in range(stage):
            strides = (1, 1)
            if i ==0 and not first_stage:
                # when layer is the first layer of this stage,
                # and is not the first block.
                # should down-sample
                strides = (2, 2)
            input = block(filter=filter,
                          strides=strides,
                          w_decay=w_decay,
                          )(input)
        return input
    return f


def _normal_block(filter, strides, w_decay):
    def f(input):
        conv = _conv_bn_relu(filter, (3, 3), s=strides, w_decay=w_decay)(input)
        residual = Conv2D(filter, (3, 3),
                       strides=1,
                       padding='same',
                       kernel_regularizer=l2(w_decay),
                       kernel_initializer=initial(seed),
                       )(conv)
        # se block
        se_out = residual
        if filter < filter_num*8 and SE_flag:
            se_out = se_block(residual)
        se_out = BatchNormalization(axis=3)(se_out)
        sum = _shortcut(input, se_out, w_decay)
        return Activation('relu')(sum)
    return f


def _bottleneck_block(filter, strides, w_decay):
    def f(input):
        conv1 = _conv_bn_relu(filter, (1, 1), s=strides, w_decay=w_decay)(input)
        conv2 = _conv_bn_relu(filter, (3, 3), s=1, w_decay=w_decay)(conv1)
        residual = Conv2D(4*filter, (1, 1),
                          strides=1,
                          padding='same',
                          kernel_regularizer=l2(w_decay),
                          kernel_initializer=initial(seed),
                          )(conv2)

        # se block
        se_out = residual
        if filter < filter_num*8 and SE_flag:
            se_out = se_block(residual)
        se_out = BatchNormalization(axis=3)(se_out)
        sum = _shortcut(input, se_out, w_decay)
        return Activation('relu')(sum)
    return f


def seresnet(input_shape=(32, 32, 3,), num_classes=10, w_decay=1e-4, depth=34, stage=None):
    if depth >= 50:
        block_function = _bottleneck_block
    else:
        block_function = _normal_block
    if stage is None:
        stage = [3, 4, 6, 3]

    # network begins
    filter = filter_num
    input = Input(input_shape)
    conv1 = _conv_bn_relu(filter=filter, k=3, s=1, w_decay=w_decay)(input)

    # repetition to build res stage
    block = conv1
    for i, r in enumerate(stage):
        block = _resblock(block_function, filter, r, w_decay, i==0)(block)
        filter *= 2

    # classifier
    avg = GlobalAvgPool2D()(block)
    output = Dense(num_classes, activation='softmax',
                   kernel_initializer=initial(seed),
                   )(avg)

    model = Model(input, output)
    model.summary()
    return model