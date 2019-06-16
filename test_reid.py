import cv2
import argparse

from libs.deep_reid import test_pairs_reid
from libs.ssm_reid import skReID



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im1', required=False, dest='im1', type=str, default='test_data/reid/3.jpg', help='Path of image 1')
    parser.add_argument('--im2', required=False, dest='im2', type=str, default='test_data/reid/4.jpg', help='Path of image 2')
    args = parser.parse_args()

    im1, im2 = cv2.imread(args.im1), cv2.imread(args.im2)
    print ("\nIdentification result for im1 and im2 \n")
    pred = test_pairs_reid(im1, im2)
    print ("\nDeep ReID: {0}\n".format(pred))

    ssm, pred = skReID(im1, im2)
    print ("Skimage match: {0}\n".format(pred))

    print ssm






