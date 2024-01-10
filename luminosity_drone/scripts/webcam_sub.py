#!/usr/bin/env python3
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
def func1(current_frame,numLabels,labels):
    contour_list = []
    cx,cy = 0,0
    for i in range(1,numLabels):
        componentMask = (labels == i).astype("uint8") * 255
        contours,hierarchy = cv.findContours(current_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_list.append(contours)
    for i in contour_list:
        cnt = i[0]
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return (cx,cy)

def callback(data):
    br = CvBridge()
    rospy.loginfo("receiving video frame")
    try:
        current_frame = br.imgmsg_to_cv2(data,"passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"Interface Error: {e}")
    current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    current_frame = cv.blur(current_frame,(10,10))
    threshold_return,current_frame = cv.threshold(current_frame,128,255,cv.THRESH_BINARY)
    kernel = np.ones((10,10),np.uint8)
    current_frame = cv.erode(current_frame,kernel,iterations = 1)
    current_frame = cv.dilate(current_frame,kernel,iterations = 1)
    (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(current_frame,4, cv.CV_32S)
    coords = func1(current_frame,numLabels,labels)
    rospy.loginfo(coords)
    cv.imshow("camera",current_frame)
    cv.waitKey(2)
def camfeed():
    rospy.init_node("video_sub_py",anonymous=True)
    rospy.Subscriber("/swift/camera_rgb/image_raw",Image,callback)
    rospy.spin()
    cv.destroyAllWindows()

if __name__ == '__main__':
    camfeed()
