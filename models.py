from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Conv1D, MaxPooling1D, Concatenate, Add, BatchNormalization, TimeDistributed, Activation

# ResNet definition
def residual_module(data, number_filters, stride, red=False, reg=0.0001):
    shortcut = data

    bn1 = BatchNormalization()(data)
    act1 = Activation('relu')(bn1)
    conv1 = Conv1D(number_filters, 3, padding='same', use_bias=False)(act1)

    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv1D(number_filters, 3, strides=stride, padding='same', use_bias=False)(act2)

    if red:
        shortcut = Conv1D(number_filters, 1, strides=stride, padding='same', use_bias=False)(act1)

    x = Add()([conv2, shortcut])

    return x

def build_resnet(stages, filters, max_states, window_size, reg=0.0001):

    inputs = Input(shape=(window_size, 1))

    x = BatchNormalization()(inputs)
    x = Conv1D(filters[0], 7, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)

    for i in range(0, len(stages)):
        stride = 1 if i == 0 else 2
        x = residual_module(x, filters[i+1], stride, red=True)
        for j in range(0, stages[i] - 1):
            x = residual_module(x, filters[i+1], 1)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    flatten = Flatten()(x)
    softmax = Dense(max_states * window_size, activation='softmax')(flatten)
    final_reshape = Reshape((window_size, max_states))(softmax)

    model = Model(inputs, final_reshape, name="ResNet CNN")

def build_split(max_states, window_size, number_filters):

    inputs = Input(shape=(window_size, 1))

    r1 = Conv1D(number_filters, 10, activation='relu')(inputs)
    r1 = MaxPooling1D(5)(r1)
    r1 = Dropout(0.5)(r1)
    r1 = Conv1D(number_filters, 10, activation='relu')(inputs)
    r1 = Conv1D(number_filters, 10, activation='relu')(inputs)
    r1 = Conv1D(number_filters, 10, activation='relu')(inputs)
    r1 = MaxPooling1D(5)(r1)

    r2 = Conv1D(number_filters, 100, activation='relu')(inputs)
    r2 = MaxPooling1D(5)(r2)
    r2 = Dropout(0.5)(r2)
    r2 = Conv1D(number_filters, 100, activation='relu')(r2)
    r2 = MaxPooling1D(5)(r2)

    x = Concatenate(axis=1)([r1, r2])

    flatten = Flatten()(x)
    softmax = Dense(max_states * window_size, activation='softmax')(flatten)
    final_reshape = Reshape((window_size, max_states))(softmax)

    model = Model(inputs, final_reshape, name="Split CNN")

def build_simple_cnn(max_states, window_size, number_filters, depth):

    new_model = Sequential(name="Simple CNN")

    for _ in range(depth):
        new_model.add(Conv1D(filters=number_filters, kernel_size=10, activation='relu'))
        new_model.add(MaxPooling1D(5))
        new_model.add(Dropout(0.3))

    new_model.add(Flatten())
    new_model.add(Dense(max_states * window_size, activation='softmax'))
    new_model.add(Reshape(window_size, max_states))

    return new_model

def build_deep_channel(max_states):

    new_model = Sequential(name="Deep Channel")

    new_model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 1, 1)))
    new_model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    new_model.add(TimeDistributed(Flatten()))

    for _ in range(3):
        new_model.add(LSTM(256, activation='relu', return_sequences=True))
        new_model.add(BatchNormalization())
        new_model.add(Dropout(0.2))

    new_model.add(Dense(max_states))
    new_model.add(Activation('softmax'))

    return new_model


