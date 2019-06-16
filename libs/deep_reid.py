import tensorflow as tf
import numpy as np
import cv2
import argparse

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '1', 'batch size for training')
tf.flags.DEFINE_string('logs_dir', 'model_data/deep_reid/', 'path to logs directory')
IMAGE_WIDTH, IMAGE_HEIGHT = 60,  160


from improved_deep_reid import network


def preprocess_pairs(images):
    split = tf.split(images, [1, 1])
    return [tf.reshape(tf.concat(split[0], axis=0), [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]



def test_pairs_reid(im1, im2):
    assert (im1 is not None), "image not found"
    assert (im2 is not None), "image not found"
    global FLAGS
    global IMAGE_WIDTH
    global IMAGE_HEIGHT

    images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    images1, images2 = preprocess_pairs(images)

    logits = network(images1, images2, FLAGS)
    inference = tf.nn.softmax(logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        #print (FLAGS.logs_dir, ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            #print('Restore model: {0}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        image1 = cv2.cvtColor(cv2.resize(im1, (IMAGE_WIDTH, IMAGE_HEIGHT)), cv2.COLOR_BGR2RGB)
        image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        image2 = cv2.cvtColor(cv2.resize(im2, (IMAGE_WIDTH, IMAGE_HEIGHT)), cv2.COLOR_BGR2RGB)
        image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        test_images = np.array([image1, image2])
        feed_dict = {images: test_images}
        prediction = sess.run(inference, feed_dict=feed_dict)

        return bool(not np.argmax(prediction[0]))






