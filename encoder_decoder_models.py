from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import datetime

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    Wg = Conv2D(num_filters, 1, padding="same")(g)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(num_filters, 1, padding="same")(s)
    Ws = BatchNormalization()(Ws)

    out = Activation("relu")(Wg + Ws)
    out = Conv2D(num_filters, 1, padding="same")(out)
    out = Activation("sigmoid")(out)

    return out * s

def decoder_block_unet(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def decoder_block_autoencoder(inputs, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = conv_block(x, num_filters)
    return x

def decoder_block_attention_unet(x, s, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(x)
    s = attention_gate(x, s, num_filters)
    x = Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def build_autoencoder(INPUT_SHAPE):
    inputs = Input(INPUT_SHAPE)

    _, p1 = encoder_block(inputs, 64)
    _, p2 = encoder_block(p1, 128)
    _, p3 = encoder_block(p2, 256)
    _, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block_autoencoder(b1, 512)
    d2 = decoder_block_autoencoder(d1, 256)
    d3 = decoder_block_autoencoder(d2, 128)
    d4 = decoder_block_autoencoder(d3, 64)

    outputs = Conv2D(3, 3, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name='AutoEncoder')
    return model

def build_unet(INPUT_SHAPE, num_classes):
    inputs = Input(INPUT_SHAPE)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block_unet(b1, s4, 512)
    d2 = decoder_block_unet(d1, s3, 256)
    d3 = decoder_block_unet(d2, s2, 128)
    d4 = decoder_block_unet(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model

def build_attention_unet(input_shape, class_count):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block_attention_unet(b1, s4, 512)
    d2 = decoder_block_attention_unet(d1, s3, 256)
    d3 = decoder_block_attention_unet(d2, s2, 128)
    d4 = decoder_block_attention_unet(d3, s1, 64)


    outputs = Conv2D(class_count, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="Attention-UNET")
    return model

def second_to_time(n):
    return str(datetime.timedelta(seconds = n))