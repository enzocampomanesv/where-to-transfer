import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Input, GlobalAveragePooling2D, Multiply, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121,EfficientNetB0

def input_processing(input_tensor):
    input_tensor = tf.concat([input_tensor, input_tensor[..., :1]], axis=-1)
    input_tensor = tf.image.resize(input_tensor, [32, 32])
    return input_tensor
def dn_feature_extractor(n_height=20, n_width=20, n_channels=2, num_classes=1, se=0):
    new_input = Input(shape=(n_height, n_width, n_channels))
    
    input_3ch_up = Lambda(input_processing)(new_input)
    
    base_model = EfficientNetB0(include_top=False, input_shape=(32, 32, 3))

    features = base_model(input_3ch_up)
    model = Model(new_input, features)
    return model
def se_block(input_tensor, ratio = 16):
    """
    Squeeze-and-Excitation block.
    """
    # Global Average Pooling
    squeeze = GlobalAveragePooling2D()(input_tensor)
    # Fully connected layer
    excitation = Dense(units=input_tensor.shape[-1] // ratio, activation='relu')(squeeze)
    excitation = Dense(units=input_tensor.shape[-1], activation='sigmoid')(excitation)
    # Rescale the feature map
    excitation = Multiply()([input_tensor, excitation])
    return excitation
def mVGG_feature_extractor(n_height=10, n_width=10, n_channels=2, num_classes=1, se = 0, dropout_rate=0.5):
    # Define the input layer
    inputs = Input(shape=(n_height, n_width, n_channels))
    base = 32
    # Block 1
    x = Conv2D(base/2, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(base/2, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Block 2
    x = Conv2D(base, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(base, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Block 3
    x = Conv2D(base*2, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(base*2, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(base*2, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=x)
    return model
    
def mVGG_regression(n_height=10, n_width=10, n_channels=2, num_classes=1, se = 0):
#     # Define the input layer
#     inputs = Input(shape=(n_height, n_width, n_channels))
    
#     # Block 1
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
#     # Block 2
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
#     # Block 3
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     if se == 1:
#         x = se_block(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = mVGG_feature_extractor(n_height, n_width, n_channels, se=se)
    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # Output Layer
    outputs = Dense(num_classes, activation='linear')(x) # Linear activation for regression
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def mrVGG_regression(n_height=10, n_width=10, n_channels=2, num_classes=1, se = 1):
    # Define the input layer
    inputs = Input(shape=(n_height, n_width, n_channels))
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if se == 1:
        x = se_block(x)
    x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Flatten and Fully Connected Layers
    x1f = Flatten()(x1)
    x2f = Flatten()(x2)
    x3f = Flatten()(x3)
    x123 = Concatenate()([x1f,x2f,x3f])
    # x = Flatten()(x123)
    x = Dense(256, activation='relu')(x123)
    # x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation='linear')(x) # Linear activation for regression
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model