##########################################################################################
# Authors: Anna Canal, Marine Collery, Sandra Picó
# Data: MTFL Dataset: http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
# Description: Code for predict smiling and gender attributes from MTFL data. Metrics: Accuracy, Precision and Recall
# Code based on the tutorial : http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
###############################################################################################
import tensorflow as tf
import numpy as np
import load_dataset

# First, pass the path of the image
smile_gender = 1# 1= smile data, 0 = gender data

test_path1= '../dataset/dataset_smile_test/'
test_path2= '../dataset/dataset_gender_test/'
image_size =  150
num_channels = 3
classes1 = ['not_smiling','smiling'] # 0 = no smiling, 1=smiling
classes2 = ['female','male'] # 0 = female, 1=male
graph1 = 'smile.ckpt.meta'
graph2 = 'gender.ckpt.meta'
batch_size = 2995

if smile_gender==1:
    classes = classes1
    test_path = test_path1
    graph_dir = graph1
else:
    classes = classes2
    test_path = test_path2
    graph_dir = graph2

data = load_dataset.read_test_set(test_path,image_size,classes)
x_batch, y_true_batch, _, cls_batch = data.test.next_batch(batch_size)

##### Chose model path ######
model_path = './Model-125/'


# Restore the saved model
session = tf.Session()
# Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(model_path+ graph_dir)
# Load the weights saved using the restore method.
saver.restore(session, tf.train.latest_checkpoint(model_path))
# Accessing the default graph which we have restored
graph = tf.get_default_graph()
# Now, let's get hold of the op that we can be processed to get the output.
y_pred = graph.get_tensor_by_name("y_pred:0")
# Feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_true_cls = tf.argmax(y_true, axis=1)


# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch,  y_true: y_true_batch}
result=session.run(y_pred, feed_dict=feed_dict_testing) # result format: [probabiliy_of_smile probability_of_non-smiling]
y_pred_cls = tf.argmax(result, axis=1)
# Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
precision, pres_op = tf.metrics.precision(y_true_cls,y_pred_cls)
recall, rec_op = tf.metrics.recall(y_true_cls, y_pred_cls)
acc = session.run(accuracy, feed_dict=feed_dict_testing)
msg = "Testing Accuracy of "+model_path5+"  {0:>6.1%}"
print(msg.format(acc))
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
pres = session.run(pres_op, feed_dict=feed_dict_testing)
rec = session.run(rec_op, feed_dict=feed_dict_testing)
print('Final Recall: %f' % (rec))
print('Final Precision: %f' % ( pres))



