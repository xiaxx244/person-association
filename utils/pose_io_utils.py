"""

"""

import cv2
import numpy as np

"""
"""

posePairs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), 
             (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
             (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]
face_idx  = [0, 1, 14, 15, 16, 17]
upper_idx = [1, 2, 3, 4, 5, 6, 7, 8, 11]
lower_idx = [8, 9, 10, 11, 12, 13]


def _round_(v):
    return int(round(v))



def readSingleImPosePair(img_path, pose_path):
    """ img_path  - 
        pose_path - 
    returns : image and list of human keypoints 
    """
    im = cv2.imread(img_path)
    file_i = open(pose_path, "r")
    stream_i = file_i.readlines()
    N_person = int(stream_i[0])

    humans=[]
    for j in xrange(1, len(stream_i)):    
        line_f = stream_i[j]
        if len(line_f) > 1:
            _, f_pts = line_f[:line_f.find(':')], line_f[line_f.find(':')+1:]
            if 'None' not in f_pts:
                f_xy = f_pts.strip().split(',')
                Mx = int(f_xy[0].strip())
                My = int(f_xy[1].strip())
                humans.append((Mx,My))
            else:
                humans.append((-1,-1))

    assert len(humans)%18==0, "** something wrong in processing posefiles"
    human_list=[]
    for i in xrange(N_person):
        human_list.append(humans[i*18:(i+1)*18])

    return im, human_list






def get_filtered_box(im, human_, mask=None, offset=[10, 10, 10, 10]):
    img_h, img_w = im.shape[:2]
    if mask is None:
        mask = range(0, len(human_))
    Mx=[i[0] for i in human_]
    My=[i[1] for i in human_]
    Mxx = [x for x in Mx if x>0 and Mx.index(x) in mask]
    Myy = [y for y in My if y>0 and My.index(y) in mask]

    # fit into the image frame
    if len(Mxx) != 0 and len (Myy) !=0:
        x1 = max(0, min(Mxx)-offset[0])
        y1 = max(0, min(Myy)-offset[1])
        x2 = min(img_w,  max(Mxx)+offset[2])
        y2 = min(img_h,  max(Myy)+offset[3])
        #{"x1":_round(x1), "y1":_round(y1), "x2":_round(x2), "y2":_round(y2)}
        return [_round_(x1), _round_(y1), _round_(x2), _round_(y2)]
    else:
        return []



def gestImPoseboxes(im, human_, annotate=False):
    """
    mapping => {0: "Nose",   1: "Neck",   2: "RShoulder", 3: "RElbow",  4: "RWrist", 5: "LShoulder", 6: "LElbow",
                7: "LWrist", 8: "RHip",   9: "RKnee",    10: "RAnkle" , 11: "LHip" , 12: "LKnee",
	       13: "LAnkle", 14: "REye", 15: "LEye",     16: "REar",    17: "LEar"}
    """
    global face_idx
    global upper_idx
    global lower_idx

    face_box = get_filtered_box(im, human_, face_idx, offset=[10, 25, 10, 10])
    upper_box = get_filtered_box(im, human_, upper_idx, offset=[10, 10, 10, 10])
    lower_box = get_filtered_box(im, human_, lower_idx, offset=[15, 15, 15, 15])
    body_box = get_filtered_box(im, human_, mask=None, offset=[15, 25, 15, 15])

    if annotate:
        im_an = np.copy(im)
        for (x,y) in human_:
            if x>0 and y>0:
                cv2.circle(im_an, (int(x), int(y)), 2, (0, 255, 0), thickness=2, lineType=4, shift=0)
        if len(face_box) != 0:
            cv2.rectangle(im_an, (face_box[0],face_box[1]), (face_box[2],face_box[3]), (0,255,0),2)
        if len(upper_box) !=0:
            cv2.rectangle(im_an, (upper_box[0],upper_box[1]), (upper_box[2],upper_box[3]), (0,0,255),2)
        if len(lower_box) !=0:
            cv2.rectangle(im_an, (lower_box[0],lower_box[1]), (lower_box[2],lower_box[3]), (255,0,0),2)
        if len(body_box)!= None:
            cv2.rectangle(im_an, (body_box[0],body_box[1]), (body_box[2],body_box[3]), (0,255,255),2)
    else:
        im_an = None

    return im_an, face_box, upper_box, lower_box, body_box



def handle_bad_corners(left, right, top, bottom, im_w, im_h):
    """
    Helper fucntion for checking if the box goes outside the image
    If so, set it to boundary 
    """
    left = np.maximum(0, left)
    top = np.maximum(0, top)
    right = np.minimum(im_w, right)
    bottom = np.minimum(im_h, bottom)    
    return (left, right, top, bottom)



def annotatePoses(im, human_list, person_ids=None, pids=None):
    """
    """
    global posePairs
    if (person_ids is None or pids is None):
        person_ids = range(0, len(human_list))
        pids = person_ids

    im_an = np.copy(im)
    img_h, img_w = im.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX; font_size = 0.8;  font_color = (0, 0, 0)

    for i, id_ in enumerate(person_ids):
        human_ = human_list[id_]
        body_box = get_filtered_box(im, human_, mask=None, offset=[15, 25, 15, 15])
        if len(body_box)!= None:
            left, right, top, bottom = body_box[0], body_box[2], body_box[1], body_box[3]
            cv2.rectangle(im_an, (left, top), (right, bottom), (0, 255, 255), 2)
            left1, top1, right1, _ = handle_bad_corners(left-2, top-40, right+2, bottom, img_w, img_h)
            cv2.rectangle(im_an, (left1, top1), (right1, top), (0, 255, 255), -1, 1)
            text_label = "ID: " + str(pids[i])
            cv2.putText(im_an, text_label, (left, top1+25), font, font_size, font_color, 2, cv2.LINE_AA)
 
        for (x,y) in human_:
            if x>0 and y>0:
                cv2.circle(im_an, (int(x), int(y)), 3, (0, 0, 255), thickness=3, lineType=8, shift=0)

        for (x,y) in posePairs:
            if human_[x][0]>0 and human_[x][1]>0 and human_[y][0]>0 and human_[y][1]>0:
                cv2.line(im_an, human_[x], human_[y], 3)

    return im_an





