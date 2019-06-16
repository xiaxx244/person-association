#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:09:01 2018

"""

import cv2
from tabulate import tabulate
import numpy as np
from os.path import join, isfile
from utils.pose_io_utils import readSingleImPosePair, get_filtered_box
import argparse
from libs.ssm_reid import skReID
from test_reid import test_pairs_reid
from libs import triple_loss_reid
from libs import aligned

def get_body_poses(img_path, pose_path):
    assert isfile(img_path), "** Image file  not found: {0}".format(img_path)
    assert isfile(pose_path), "** Pose file  not found: {0}".format(fname)
    im, human_list= readSingleImPosePair(img_path, pose_path)

    body_boxes=[]
    for human_ in human_list:
        body_box = get_filtered_box(im, human_, offset=[15, 25, 15, 15])
        body_boxes.append(body_box)
    return im, human_list, body_boxes



def getREID(p1, p2):
    res1 = test_pairs_reid(p1, p2)
    _, res2 = skReID(p1, p2)
    res3 = triple_loss_reid.reid()
    res4 = aligned.reid("model_data/query/q1.jpg", "model_data/query/q2.jpg")
    return [res1, res2, res3, res4]

#print method for four different methods using a table
def print_table(total_pred):
    deep_reid=["deep_reid"]
    skReIDl=["skReID"]
    triple=["triplet_loss"]
    aligned_reid=["aligned_reid"]
    header=['']
    # print all predictions in the prediction list
    for i in range(len(total_pred)):
        pred,count=total_pred[i]
        for k in pred:
            idn,pred1,pred2,pred3,pred4=k
            deep_reid.append(pred1)
            skReIDl.append(pred2)
            triple.append(pred3)
            aligned_reid.append(pred4)
            header.append('pair:'+'('+str(i)+','+str(idn)+')')
            k=len(pred)
    #default the person which has not been called person re-identification to false
        while k<count:
            deep_reid.append(0)
            skReIDl.append(0)
            triple.append(0)
            aligned_reid.append(0)
            header.append('pair:'+'('+str(i)+','+str(k)+')')
            k=k+1
    print("result table for four different methods below:")
    print tabulate([deep_reid,skReIDl,triple,aligned_reid], headers=header)



def associate_pairs(img_path1, pose_path1, img_path2, pose_path2):

    im1, human_list1, body_boxes1 = get_body_poses(img_path1, pose_path1)
    im2, human_list2, body_boxes2 = get_body_poses(img_path2, pose_path2)

    persons_i, persons_j = [], []
    i=0; exclude=[];total_pred=[]
    while i< len(body_boxes1):
        pred=[]
        body_i = body_boxes1[i]
        person_i = im1[body_i[1]:body_i[3],body_i[0]:body_i[2],:]
        cv2.imwrite("model_data/query/q1.jpg",person_i)
        for j in xrange(len(body_boxes2)):
            if j not in exclude:
                body_j = body_boxes2[j]
                person_j = im2[body_j[1]:body_j[3], body_j[0]:body_j[2],:]
                cv2.imwrite("model_data/query/q2.jpg",person_j)
                [pred1,pred2,pred3,pred4]=getREID(person_i,person_j)
                count=[pred1,pred2,pred3,pred4].count(True)
                pred.append([j,pred1,pred2,pred3,pred4])
                if count>=2 :
                    persons_i.append(i)
                    persons_j.append(j)
                    exclude.append(j)
                    break
            else:
                pred.append([j,0,0,0,1])
        total_pred.append((pred,len(body_boxes1)))
        #print_table(pred,i,len(body_boxes1))
        i=i+1
    print_table(total_pred)
    assert len(persons_i)==len(persons_j), "** association failed"
    return im1, im2, human_list1, human_list2, persons_i, persons_j





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im1', required=False, dest='im1', type=str, default='test_data/pose/pool/5192.jpg')
    parser.add_argument('--pose1', required=False, dest='pose1', type=str, default='test_data/pose/pool/5192.txt')
    parser.add_argument('--im2', required=False, dest='im2', type=str, default='test_data/pose/pool/5504.jpg')
    parser.add_argument('--pose2', required=False, dest='pose2', type=str, default='test_data/pose/pool/5504.txt')
    parser.add_argument('--viz', required=False, dest='viz', type=bool, default=True)
    args = parser.parse_args()

    im1, im2, humans1, humans2, ids1, ids2 = associate_pairs(args.im1, args.pose1, args.im2, args.pose2)
    print ("The pairs are {0}".format(zip(ids1, ids2)))

    if args.viz:
        from utils.pose_io_utils import annotatePoses
        pids = range(0, len(ids1))
        im_an1 = annotatePoses(im1, humans1, ids1, pids)
        im_an2 = annotatePoses(im2, humans2, ids2, pids)
        combo = np.concatenate((im_an1, im_an2), axis=1)
        cv2.imwrite("etc/combo3.png",combo)
        #cv2.imshow("test", combo)
        #cv2.waitKey(2000)
