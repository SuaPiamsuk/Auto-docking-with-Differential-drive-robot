#!/usr/bin/env python
# # license removed for brevity
# import rospy
# from std_msgs.msg import String

# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass



import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class FisheyeUndistortNode:
    def __init__(self):
        rospy.init_node('fisheye_undistort_node', anonymous=True)

        # Subscribe to fisheye camera image topic
        self.image_sub = rospy.Subscriber('/camera/fisheye2/image_raw', Image, self.image_callback)

        # Subscribe to camera info topic
        self.camera_info_sub = rospy.Subscriber('/camera/fisheye2/camera_info', CameraInfo, self.camera_info_callback)

        # Publisher for undistorted image
        self.undistorted_img_pub = rospy.Publisher('/camera/undistorted/image_raw', Image, queue_size=10)

        self.undistorted_info_pub = rospy.Publisher('/camera/undistorted/camera_info', CameraInfo, queue_size=10)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.DIM = None

        self.camera_info_msg = CameraInfo()


    def camera_info_callback(self, msg):
        # Store camera matrix and distortion coefficients
        self.K = np.reshape(msg.K, [3, 3])
        self.D = msg.D[:4]
        self.DIM = (msg.width, msg.height)

        #store msg for re-publish info
        self.camera_info_msg = msg
        self.camera_info_msg.roi.do_rectify = True
        self.camera_info_msg.distortion_model = 'plumb_bob'

        # # Unsubscribe from the camera info topic after receiving it once
        self.camera_info_sub.unregister()


    def image_callback(self, data):
        print(self.camera_info_msg)

        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_32FC1)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        output_msg = self.bridge.cv2_to_imgmsg(undistorted_img, 'mono8')

        current_time = rospy.Time.now()

        output_msg.header.frame_id = 'camera_fisheye2_optical_frame'
        output_msg.header.stamp = current_time
        self.camera_info_msg.header.stamp = current_time

        self.undistorted_img_pub.publish(output_msg)
        self.undistorted_info_pub.publish(self.camera_info_msg)

        # try:
        #     # Convert ROS Image message to OpenCV image
        #     cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        #     # Undistort the image using camera information
        #     if self.camera_matrix is not None and self.distortion_coefficients is not None:
        #         undistorted_image = cv2.fisheye.undistortImage(cv_image, self.camera_matrix, self.distortion_coefficients)

        #         # Convert the undistorted image back to ROS Image message
        #         undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_image, 'mono8')

        #         # Publish the undistorted image
        #         self.undistorted_pub.publish(undistorted_msg)

        # except CvBridgeError as e:
        #     rospy.logerr(e)

if __name__ == '__main__':
    try:
        fisheye_undistort_node = FisheyeUndistortNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

