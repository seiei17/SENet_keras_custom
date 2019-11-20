# runfile
import tensorflow
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from math import ceil
import numpy as np
import os

from SEResNet import seresnet
from CifarGenerator import CifarGen

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

num_classes = 10
path = '../../database/cifar{}/'.format(num_classes)
checkpoint_path = './history/checkpoint.h5'
txt_path = './history/accuracy.txt'

resume = False
epochs = 200
val_size = 0.05
batch_size = 32
lr = 1e-3
w_decay = 1e-4

depth = 50
stage = [3, 4, 6, 3]
# stage = None

train_steps = ceil(50000 * (1 - val_size) / batch_size)
valid_steps = ceil(5000 * val_size / batch_size)

model = seresnet((32, 32, 3,), num_classes, w_decay, depth, stage)
gen = CifarGen(path, batch_size, num_classes)


def lr_reducer(epochs):
    new_lr = lr
    if epochs > 180:
        new_lr = .5e-3
    elif epochs > 160:
        new_lr = 1e-3
    elif epochs > 120:
        new_lr = 1e-2
    elif epochs > 80:
        new_lr = 1e-1
    return new_lr
lr_reduce_scheduler = LearningRateScheduler(lr_reducer, verbose=1)
checkpoint = ModelCheckpoint(checkpoint_path, 'val_accuracy', 1, True, True)
lr_reduce_Plateau = ReduceLROnPlateau('val_loss', np.sqrt(0.1), 5, 1, min_lr=.5e-6)

if resume:
    model.load_weights(checkpoint_path)

x_train, y_train = gen.train_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size)
print('train data shape is:', x_train.shape[0])
print('validation data shape is:', x_val.shape[0])

model.compile(Adam(lr), categorical_crossentropy, ['accuracy'])
history = model.fit_generator(gen.train_gen(x_train, y_train),
                              steps_per_epoch=train_steps,
                              epochs=epochs,
                              verbose=1,
                              callbacks=[lr_reduce_Plateau, lr_reduce_scheduler, checkpoint],
                              validation_data=gen.valid_gen(x_val, y_val),
                              validation_steps=valid_steps)

tr_acc = np.array(history.history['accuracy']).reshape((-1, 1))
va_acc = np.array(history.history['val_accuracy']).reshape((-1, 1))
np.savetxt(txt_path, np.concatenate([tr_acc, va_acc], axis=1), fmt='%.5f')
