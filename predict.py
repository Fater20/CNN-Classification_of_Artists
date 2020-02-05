import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import shutil

image_size=64
num_channels=3
images = []

test_path = 'C:/Users/dell/Desktop/final1/train_data_final/test_set'
path = os.path.join(test_path, '*g')
files = glob.glob(path)
for fl in files:

    image = cv2.imread(fl)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./model/painting.ckpt-7990.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, './model/painting.ckpt-7990')

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 11))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    # dog [1 0]
    res_label = ['canaletto','claude_monet','george_romney','jmw_turner','john_robert_cozens','paul_cezanne','paul_gauguin','paul_sandby','peter_paul_rubens','rembrandt','richard_wilson']
    name = res_label[result.argmax()]
    print( name )
    name_path= 'C:/Users/dell/Desktop/final1/result/' + name
    shutil.copy(fl, name_path )
    images = []
    