import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('fer2013.csv')

pixels = data['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
pixels = np.array(pixels.tolist()).reshape(-1, 48, 48, 1)
emotions = to_categorical(data['emotion'])

x_train, x_test, y_train, y_test = train_test_split(pixels, emotions, test_size=0.2, random_state=42)

def FER_Model(input_shape=(48, 48, 1)):
    visible = Input(shape=input_shape, name='input')
    num_classes = 7
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name='drop1_1')(pool1_1)  # the 2-nd block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_3')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_3)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_3)
    drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)  # the 3-rd block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1')(drop2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_3')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_4')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), name='pool3_1')(conv3_4)
    drop3_1 = Dropout(0.3, name='drop3_1')(pool3_1)  # the 4-th block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_3')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_4')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name='pool4_1')(conv4_4)
    drop4_1 = Dropout(0.3, name='drop4_1')(pool4_1)
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_3')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_4')(conv5_3)
    conv5_3 = BatchNormalization()(conv5_3)
    pool5_1 = MaxPooling2D(pool_size=(2, 2), name='pool5_1')(conv5_4)
    drop5_1 = Dropout(0.3, name='drop5_1')(pool5_1)  # Flatten and output
    flatten = Flatten(name='flatten')(drop5_1)
    ouput = Dense(num_classes, activation='softmax', name='output')(flatten)  # create model
    model = Model(inputs=visible, outputs=ouput)
    print(model.summary())

    return model

model = FER_Model()

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))

model.save('Emoti0.2.keras')

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
