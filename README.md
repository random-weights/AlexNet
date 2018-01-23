# AlexNet
TensorFlow implementation of Alexnet 2012 (IMAGENET)

## See branch layers-api for implementation of same network in tf.layers highlevel api.
In this particular example the api isn't much useful, infact it made the code a little longer. But for large heterogenous networks with no uniform order
of convolution and pooling layers, this make sense.

Perhaps wrapping tf.layers.conv2d inside a helper function will eliminate lengthy expression.
