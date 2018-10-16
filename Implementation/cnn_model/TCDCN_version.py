##########################################################################################
# Authors: Anna Canal, Marine Collery, Sandra Picó
# Data: MTFL Dataset: http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
# Description: Code for training a CNN with MTFL dataset, based on TCDCN, taking smiling and gender attributes
# Code based on the tutorial : http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
###############################################################################################

import tensorflow as tf
import load_dataset

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


##################################   INIT     ################################
batch_size = 100
learning_rate = 1e-3 #1e-4
validation_size = 0.2 # 20% of the data will automatically be used for validation
img_size = 150
num_channels = 3 #RGB (if grey scale, num_channels=1)

smile_gender = 1 # 1= smile data, 0 = gender data
classes1 = ['not_smiling','smiling'] # 0 = no smiling, 1=smiling
classes2 = ['female','male'] # 0 = no smiling, 1=smiling
num_classes = len(classes1)
train_path1='../dataset/dataset_smile_cropped/'
model_dir1= './smile-model-tcdcn/smile.ckpt'
train_path2='../dataset/dataset_gender_cropped/'
model_dir2= './gender-model-tcdcn/gender.ckpt'
#load all the training and validation images and labels into memory
data1 = load_dataset.read_train_sets(train_path1, img_size, classes1, validation_size=validation_size)
data2 = load_dataset.read_train_sets(train_path2, img_size, classes2, validation_size=validation_size)

if smile_gender==1:
    data= data1
    model_dir= model_dir1
    train_path = train_path1
else:
    data= data2
    model_dir= model_dir2
    train_path = train_path2


print("Complete reading input data.")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
session = tf.Session()
# Define the graph
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
W = tf.Variable(tf.zeros([150 * 150, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

# Network parameters
filter_size_conv1 = 5
num_filters_conv1 = 20
filter_size_conv2 = 5
num_filters_conv2 = 48
filter_size_conv3 = 3
num_filters_conv3 = 64
filter_size_conv4 = 3
num_filters_conv4 = 80
fc_layer_size = 256

################################ FUNCTIONS ######################3
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,num_input_channels,conv_filter_size,num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    ## Create the convolutional layer
    layer = tf.nn.conv2d(input=input,  filter=weights, strides=[1, 1, 1, 1],  padding="VALID")
    layer = tf.nn.bias_add(layer, biases)
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    return layer

def create_pool_layer(layer):
    ##max-pooling.
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, model_dir)

    total_iterations += num_iteration


################################## NETWORK ####################################

layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
layer_pool1 = create_pool_layer(layer_conv1)

layer_conv2 = create_convolutional_layer(input=layer_pool1, num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)
layer_pool2 = create_pool_layer(layer_conv2)

layer_conv3 = create_convolutional_layer(input=layer_pool2, num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3)
layer_pool3 = create_pool_layer(layer_conv3)

layer_conv4 = create_convolutional_layer(input=layer_pool3, num_input_channels=num_filters_conv3,
                                         conv_filter_size=filter_size_conv4, num_filters=num_filters_conv4)
layer_pool4 = create_pool_layer(layer_conv4)

layer_flat = create_flatten_layer(layer_pool4 )

layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,  use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
total_iterations = 0
saver = tf.train.Saver()

train(num_iteration=3000)

