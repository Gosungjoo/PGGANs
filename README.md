# PGGANs

Ref : https://github.com/tkarras/progressive_growing_of_gans

I made the pggan function using TensorFlow as a class layer.

#  minibatch STD


x = MinibatchSTDDEV()(input_d)
-------------------------------------------------------------------------
#  conv layer 3x3 (use Weight Normalization)


x = tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(3, 3, strides=1,
        padding='same',kernel_initializer=kernel_initializer, bias_initializer='zeros'))(x)
    
---------------------------------------------------------------------------
#  Leaky ReLU


x = LeakyReLU()(x)
