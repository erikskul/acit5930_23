#!/usr/bin/env python

import roslib
roslib.load_manifest('terrain_characterizer')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

import time

from matplotlib import pyplot as plt

program_starts = time.time()
counter = 0

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/dyret/sensor/camera/depth",Image,self.callback)

  def callback(self,data):
    global counter

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
    except CvBridgeError as e:
      print(e)

    (rows,cols) = cv_image.shape

    width = int(cols * 0.75)
    height = rows
    
    # Crop image
    cv_image_cropped = cv_image[rows/2 - height/2 : rows/2 + height/2, cols/2 - width/2 : cols/2 + width/2]
    cv_image_cropped[cv_image_cropped == 65535] = 0

    # Discard the top percent of points
    cv_image_cropped[cv_image_cropped > np.percentile(cv_image_cropped, 99)] = 0

    points = []
    distances = []

    (rows, cols) = cv_image_cropped.shape

    cv_image_normalized = cv_image_cropped
    cv_image_normalized[cv_image_normalized != 0] = np.subtract(cv_image_normalized[cv_image_normalized != 0],
                                                                np.min(cv_image_normalized[cv_image_normalized != 0]))
    cv_image_normalized = np.floor_divide(cv_image_normalized, np.max(cv_image_normalized) / 255.0)

    # Convert to color and display
    cv_image__8bit = np.array(cv_image_normalized, dtype=np.dtype('uint8'))
    cv_image_colored = cv2.applyColorMap(cv_image__8bit, cv2.COLORMAP_JET)
    cv2.imshow("Image window", cv_image_colored)
    cv2.waitKey(3)

def main(args):
  ic = image_converter()
  rospy.init_node('depth_viewer')
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

