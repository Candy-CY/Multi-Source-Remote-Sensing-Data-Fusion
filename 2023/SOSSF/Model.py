from keras.models import Model
import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Conv2DTranspose, BatchNormalization,\
    Activation, Dropout, Softmax, Permute, Lambda, Reshape, Conv1D, Conv3D, add, dot,Subtract, Add, Concatenate
import keras
import keras.backend as K
from sklearn import preprocessing
import numpy as np

def Ups(input_tensor):
    lambda_D2S = Lambda(lambda x: tf.depth_to_space(x, 2))
    [b, h, w, c] = input_tensor.get_shape().as_list()
    Conv_F = Conv2D(c*2, kernel_size=(3, 3), padding='same',use_bias=False)(input_tensor)
    output = lambda_D2S(Conv_F)
    output = Activation("relu")(output)
    return output
 

    
def Downs(input_tensor):
    lambda_S2D = Lambda(lambda x: tf.space_to_depth(x, 2))
    [b, h, w, c] = input_tensor.get_shape().as_list()
    Conv_F = Conv2D(c//2, kernel_size=(3, 3), padding='same',use_bias=False)(input_tensor)
    output = lambda_S2D(Conv_F)
    output = Activation("relu")(output)
    return output    
    
    
def Residual(input_tensor, filters, kernel_size):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """

    x1 = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x1 = Activation("relu")(x1)

    x2 = Conv2D(filters, kernel_size, padding='same')(x1)

    weight_1 = Lambda(lambda x:x*0.1)
    weight_x2 = weight_1(x2)
    output = keras.layers.add([input_tensor, weight_x2])
    return output


def SCAB(input_m,input_s):
    [b, h, w, c] = input_m.get_shape().as_list()
    input_ADD = keras.layers.add([input_m, input_s])
    
    m1 = Conv2D(c, (1, 1), activation='relu', padding='same')(input_m)
    m1 = GlobalAveragePooling2D()(m1)
    m1 = Reshape((1, c))(m1)
    m1 = Softmax(axis=2)(m1)# [N, 1, c]
    m1 = Permute((2, 1))(m1)  #[N, c, 1]
    m2 = Reshape((h * w, c))(input_ADD)
    
    lambda_batchdot = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    x1 = lambda_batchdot([m2, m1]) # b*hw * 1
    x1 = Reshape((h , w, 1))(x1)
    x1_trans = Conv2D(1, (3, 3), activation='relu', padding='same')(x1)
    x1_trans = Conv2D(1, (3, 3), activation=None, padding='same')(x1_trans)
    
    
    s1 = Conv2D(1, (1, 1), activation='relu', padding='same')(input_s)
    s1 = Reshape((h * w, 1))(s1)
    s1 = Softmax(axis=1)(s1)# [N, 1, c]
    
    s2 = Reshape((h * w, c))(input_ADD)
    s2 = Permute((2, 1))(s2) 
    x2 = lambda_batchdot([s2, s1]) # b*c * 1
    x2 = Permute((2, 1))(x2) 
    lambda_expend = Lambda(lambda x: tf.expand_dims(x, axis=1))
    x2 = lambda_expend(x2)# b*1 * 1* c
    x2_trans = Conv2D(c//4, (1, 1), activation='relu', padding='same')(x2)
    x2_trans = Conv2D(c, (1, 1), activation=None, padding='same')(x2_trans)
    out = keras.layers.add([input_ADD, x1_trans,x2_trans])
    return out
    
def AddFusion(input_m,input_s):
    [b, h, w, c] = input_m.get_shape().as_list()
    input_ADD = keras.layers.add([input_m, input_s])  
    out = input_ADD
    return out   

def model_arch(input_rows=512, input_cols=512, num_of_channels=4, num_of_classes=1):
    inputs_a = Input((input_rows, input_cols, 6))  # 256,256,4   # reference cloudless image
    inputs_b = Input((input_rows, input_cols, 2))                # cloudy image to be detected
    
    basenum = 64
    conv1 = Conv2D(basenum, (3, 3), activation='relu', padding='same')(inputs_a)
    conv1_ = Downs(conv1)

    conv2 = Conv2D(basenum*2, (3, 3), activation='relu', padding='same')(conv1_)
    conv2_ = Downs(conv2)                           # 64,64,64

    conv3 = Conv2D(basenum*4, (3, 3), activation='relu', padding='same')(conv2_)
    conv3_ = Downs(conv3)

    conv4 = Conv2D(basenum*8, (3, 3), activation='relu', padding='same')(conv3_)

    S_conv1 = Conv2D(basenum, (3, 3), activation='relu', padding='same')(inputs_b)
    S_conv1_ = Downs(S_conv1)

    S_conv2 = Conv2D(basenum*2, (3, 3), activation='relu', padding='same')(S_conv1_)
    S_conv2_ = Downs(S_conv2)                           # 64,64,64

    S_conv3 = Conv2D(basenum*4, (3, 3), activation='relu', padding='same')(S_conv2_)
    S_conv3_ = Downs(S_conv3)

    S_conv4 = Conv2D(basenum*8, (3, 3), activation='relu', padding='same')(S_conv3_)

    conv4_fusion = SCAB( conv4 , S_conv4 )
    conv3_fusion = AddFusion( conv3 , S_conv3 )
    conv2_fusion = AddFusion( conv2 , S_conv2 )
    conv1_fusion = AddFusion( conv1 , S_conv1 )

    convT9 = Ups( conv4_fusion )
    up9 = concatenate([conv3_fusion, convT9], axis=3)
    conv9 = Conv2D(basenum*4, (3, 3), activation='relu', padding='same')(up9)
    conv9_ = Residual(conv9, basenum*4, 3)
    conv9_ = Residual(conv9_, basenum*4, 3)
    conv9_ = Residual(conv9_, basenum*4, 3)
    conv9_ = Residual(conv9_, basenum*4, 3)

    convT10 = Ups( conv9_ )
    up10 = concatenate([conv2_fusion, convT10], axis=3)
    conv10 = Conv2D(basenum*2, (3, 3), activation='relu', padding='same')(up10)
    conv10_ = Residual(conv10, basenum*2, 3)
    conv10_ = Residual(conv10_, basenum*2, 3)
    conv10_ = Residual(conv10_, basenum*2, 3)
    conv10_ = Residual(conv10_, basenum*2, 3)

    convT11 = Ups( conv10_ )
    up11 = concatenate([conv1_fusion, convT11], axis=3)
    conv11 = Conv2D(basenum, (3, 3), activation='relu', padding='same')(up11)
    conv12 = Conv2D(6, (3, 3), activation=None, padding='same')(conv11)
    
    return Model(inputs=[inputs_a,inputs_b], outputs=[conv12])


