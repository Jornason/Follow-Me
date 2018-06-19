# Follow Me Project

## Introduction

In this project I build and train fully convolutional neural network, that makes segmentation. This network can perform tracking of a target person in the simulation. 

## Network architecture

### Fully Convolutional Network (FCN)

FCN consists of 2 parts connected with 1x1 Convolution:

1. Encoder

   Encoder extracts useful features from the images. This is just a set  of convolutional layers. Each layer of the encoder “squeezes” information form the input image  into smaller amount of variables. In classical Convolutional network flatten the output by connecting it  to the fully connected layer. Which leads to the spatial information  loss. 

   **1x1 Convolution**

   In order to keep spatial information we can use 1x1 convolution. It  helped in reducing the dimensionality of the layer and avoid flattening  of the output information.

2. Decoder

   instead of using fully connected layers, we will use decoder. It upsamples/reconstruct information that was compressed by encoder and tries to reconstract the original image. 

**Network Structure**

This 3 encoder-decoder architecture was selected through testing during the Semantic Segmentation lab. With basic parameters, the 3 encoder/decoder model get very close to the target score (0.37),still needed some improvements to pass the project. I can play a little bit more with the  hyper-parameters.



![mark](http://owj75jsw8.bkt.clouddn.com/blog/180619/B2a16L577A.png?imageslim)

```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_1 = encoder_block(inputs, 32, 2)
    encoder_2 = encoder_block(encoder_1, 64, 2)
    encoder_3 = encoder_block(encoder_2, 128, 2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder_3, 128, kernel_size=1, strides=1)
        
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_1 = decoder_block(conv_layer, encoder_2, 128)
    decoder_2 = decoder_block(decoder_1, encoder_1, 64)
    decoder_3 = decoder_block(decoder_2, inputs, 32)   
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder_3)
```







![mark](http://owj75jsw8.bkt.clouddn.com/blog/180619/8IlE5BgJb1.png?imageslim)

## Hyper Paramaters

In the code section below, the chosen network parameters can be seen. The hyperparameters were modified by  iterations until the final score of 0.4096 was achieved.

* Epoch: I have used 10,20 epochs for the training of the network. 10 reached a  good result but not good enough for the submission, 20 epochs show no siginificent improments but may result overfitting.
* Learning rate: the first training step was done with a learning rate of 0.001,  and the second training step was done with a learning rate of 0.0001.
* Batch size: I have used the same size I used for the lab, as analyzing the  images in blocks of 20 sounds reasonable to have a wide-range of different  features to analyze without having memory problems
* Steps per epoch: I also adapted this value to the batch size, and as our  training set has approximately 4100 pictures, for each bach, we just need to  iterate over 207 blocks of 20 pictures.

```
learning_rate = 0.0001
batch_size = 20
num_epochs = 10
steps_per_epoch = 207
validation_steps = 50
workers = 4
```

## Future work 

To improve NN results I can do following:

1. Gather more data, especially in performing bad case, eg the target is far away in the picture.
2. add more layers
3. Use a variable learning rate
4. I can also use different NN types.

## Model

 The final weights are in the data/weights path and the 'model_weights_fixed.h5'  and 'config_model_weights_fixed.h5' are the final weight files. 



















