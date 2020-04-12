from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
import cv2
import math
from numpy import array
from sklearn.cluster import KMeans
from io import StringIO
from PIL import Image
import math
import matplotlib.pyplot as plt
import time
from utils import visualization_utils as vis_util
from utils import label_map_util
from sklearn.feature_extraction import image
from sklearn.cluster import DBSCAN
from multiprocessing.dummy import Pool as ThreadPool
from matplotlib import pyplot as plt
MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE =0.9

PATH_TO_LABELS = 'detection3/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'test_images'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'output5'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def getREID(p1, p2):
    res1 = test_pairs_reid(p1, p2)
    _, res2 = skReID(p1, p2)
    res3 = triple_loss_reid.reid()
    res4 = aligned.reid("model_data/query/q1.jpg", "model_data/query/q2.jpg")
    return [res1, res2, res3, res4]
def association_pairs(total_p):
    i=1
    im1=cv2.imread(total_p[0][0][0])
    vec3=[]
    for m in range(len(total_p[0])):
        vec3.append(total_p[0][m][0],total_p[0][m][1],m)
    while i<len(total_p):
        exclude=[]
        im2=cv2.imread(total_p[i][0][0])
        for j in range(len(total_p[i])):
            body_i=total_p[i][j][1]
            person_i = im1[body_i[1]:body_i[3],body_i[0]:body_i[2],:]
            for k in range(len(total_p[0])):
                body_j=total_p[0][k][1]
                person_j = im2[body_j[1]:body_j[3], body_j[0]:body_j[2],:]
                cv2.imwrite("model_data/query/q2.jpg",person_j)
                [pred1,pred2,pred3,pred4]=getREID(person_i,person_j)
                count=[pred1,pred2,pred3,pred4].count(True)
                pred.append([j,pred1,pred2,pred3,pred4])
                if count>=2 :
                    #persons_i.append(i)
                    vec3.append(total_p[i][j][0],total_p[i][j][1],k)
                    break
    return vec3

def detect_objects(image_path):
    image = Image.open(image_path)
    (im_width,im_height)=image.size
    image2 = cv2.imread(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    total=[]
    person=[]
    for i in range(int(num[0])):
        if classes[0][i]!=2:
            feature=[]
            ymin=int(boxes[0][i][0]*im_height)
            ymax=int(boxes[0][i][2]*im_height)
            xmin=int(boxes[0][i][1]*im_width)
            xmax=int(boxes[0][i][3]*im_width)
            person.append((image_path,(ymin,xmin,ymax,xmax)))
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8)
    fig = plt.figure()
    fig.set_size_inches(16, 9)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image_np, aspect = 'auto')
    plt.savefig('output/{}'.format(image_path), dpi = 62)
    plt.close(fig)
    return person

TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
total_p=[]
# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
names=['Emma','Liam','Noah','Olivia',
'William','Ava', 'James'
'Isabella','Logan','Sophia','Benjamin',
'Mia','Mason','Charlotte',
'Elijah','Amelia',
'Oliver','Evelyn',
'Jacob','Abigail']
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        #count=0
        for image_path in TEST_IMAGE_PATHS:
            total_p.append(detect_objects(image_path))
            #count=count+1
        vec3=association_pairs(diver,max)
        for (image_path,(ymin,xmin,ymax,xmax),id) in vec3:
            image=cv2.imread(image_path)
            cv2.rectangle(image,(xmin,ymax),(xmax,ymin),(0,255,0),2)
            cv2.putText(image, names[id], (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            cv2.imwrite(image_path,image)
