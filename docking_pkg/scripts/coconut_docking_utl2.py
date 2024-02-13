#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import geometry_msgs.msg

from std_msgs.msg import String
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray

from docking_pkg.srv import docking_srv        

import tf
import os
import yaml
# import geometry_msgs.msg import TransformStamped

import numpy as np

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

class DemoNode():
    def __init__(self):
        print("entering constructor")

        file_path = os.path.join(os.path.dirname(__file__), "../config/coconut_docking_utl2.yaml")

        with open(file_path, 'r') as file:
            self.config_data = yaml.safe_load(file)
        
        self.cmd_vel_topic = self.config_data['topic']['cmd_vel']
        self.odom_topic = self.config_data['topic']['odom']

        self.odom_frame = self.config_data['frame']['odom']
        self.base_footprint = self.config_data['frame']['base_footprint']

        self.apriltag_left = self.config_data['apriltag']['left']
        self.apriltag_middle = self.config_data['apriltag']['middle']
        self.apriltag_right = self.config_data['apriltag']['right']

        # Declare angle metrics in radians
        self.heading_tolerance_ = self.config_data['control']['heading_tolerance']
        self.yaw_goal_tolerance = self.config_data['control']['yaw_goal_tolerance']

        # Declare linear and angular velocities
        self.linear_velocity_max = self.config_data['control']['linear_velocity_max']  # meters per second
        self.angular_velocity_max = self.config_data['control']['angular_velocity_max'] # radians per second

        # Declare distance metrics in meters
        self.distance_goal_tolerance_first_predocking_ = self.config_data['control']['distance_goal_tolerance_first_predocking']
        self.distance_goal_tolerance_next_predocking_ = self.config_data['control']['distance_goal_tolerance_next_predocking']
        self.distance_goal_tolerance_last_predocking = self.config_data['control']['distance_goal_tolerance_last_predocking']

        self.yaw_adjust_disable_by_distance_x = self.config_data['control']['yaw_adjust_disable_by_distance_x']
        self.heading_error_threshold = self.config_data['control']['heading_error_threshold']

        self.palabola_alpha = self.config_data['control']['palabola_params']['alpha']
        self.palabola_dist = self.config_data['control']['palabola_params']['max_dist_to_deatect_all_tags']
        self.palabola_beta = self.palabola_alpha / (-1.0*(self.palabola_dist**2))

        self.delay_before_start = self.config_data['docking']['delay_before_start']

        self.min_dist_to_deatect_all_tags = self.config_data['docking']['min_dist_to_deatect_all_tags']

        self.first_pre_docking_distance = self.config_data['docking']['first_pre_docking_distance']
        self.docking_distance = self.config_data['docking']['docking_distance']
        self.docking_direction = self.config_data['docking']['docking_direction']

        self.kp_angular = self.config_data['control']['kp_angular']
        self.kp_linear = self.config_data['control']['kp_linear']

        
    
        ###################################################################################

        # Publisher
        self.cmd_vel = rospy.Publisher(self.cmd_vel_topic, geometry_msgs.msg.Twist, queue_size=10)

        # Subscriber
        self.subscription = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.listener_callback)
        self.subscription  # prevent unused variable warning
        self.apriltag_msg = None
        
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.odom_sub  # prevent unused variable warning

        # Service
        self.srv = rospy.Service('set', docking_srv, self.srv_callback)
        
        print("creating a publisher")

        # Timer
        self.duration_time = 0.08
        self.timer = rospy.Timer(rospy.Duration(self.duration_time), self.demo_callback)
        print("timer called")

        # TF
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.static_pre_docking_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.dynamic_pre_docking_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_odom_to_robot = None
        self.tf_tag25_to_robot = None

        # Kalman filter params 1
        self.K = 0
        self.x = 0
        self.P = 0
        self.P_pre = 0
        self.R = 0.00000015523403536447
        self.C = 1
        self.Q = 0.00009

        # Kalman filter params 2
        self.K1 = 0
        self.x1 = 0
        self.P1 = 0
        self.P_pre1 = 0
        self.R1 = 0.00008339191740914060
        self.C1 = 1
        self.Q1 = 0.00009

        # Declare angle metrics in radians
        

        # Declare linear and angular velocities
        



        # Initial values
        self.first_point = True
        self.docking_state = 'init'
        self.time_ = 0.0
        self.current_direction = 0.0
        self.odom_current = PoseStamped()
        self.current_yaw = None
        self.desired_x = None
        self.desired_y = None
        self.desired_theta = None
        
        # Declare distance metrics in meters
        self.distance_goal_tolerance = None
        self.reached_distance_goal = False 
        

    def srv_callback(self,msg):
        self.docking_state = 'init'
        self.first_point = True
        self.direction = 0.0
        output = 'start docking'
        print("service called & robot moving")
        return output

    def odom_callback(self, msg):
        self.odom_current = PoseStamped()
        self.odom_current.pose = msg.pose.pose

        _, _, current_yaw = euler_from_quaternion(self.odom_current.pose.orientation.x, self.odom_current.pose.orientation.y, self.odom_current.pose.orientation.z, self.odom_current.pose.orientation.w)
        self.current_yaw = round(current_yaw,3)

        self.odom_current.pose.position.x = round(self.odom_current.pose.position.x,3)
        self.odom_current.pose.position.y = round(self.odom_current.pose.position.y,3)
    
    def get_distance_to_goal(self):
        """
        Get the distance between the current x,y coordinate and the desired x,y coordinate
        The unit is meters.
        """
        distance_to_goal = math.sqrt(math.pow(self.desired_x - self.odom_current.pose.position.x, 2) + math.pow(
            self.desired_y - self.odom_current.pose.position.y, 2))
        return distance_to_goal

    def get_heading_error(self):
        """
        Get the heading error in radians
        """
        delta_x = self.desired_x - self.odom_current.pose.position.x
        delta_y = self.desired_y - self.odom_current.pose.position.y
        desired_heading = math.atan2(delta_y, delta_x) 
        
        if self.current_direction > 0.0:
            heading_error = desired_heading - self.current_yaw # forward only
        elif self.current_direction < 0.0:
            heading_error = desired_heading - self.current_yaw - math.pi # backward only
        
        # Make sure the heading error falls within -PI to PI range
        if (heading_error > math.pi):
            heading_error = heading_error - (2 * math.pi)
        if (heading_error < -math.pi):
            heading_error = heading_error + (2 * math.pi)
        
        return heading_error
    
    def get_radians_to_goal(self):
        """
        Get the yaw goal angle error in radians
        """   
        yaw_goal_angle_error = self.desired_theta - self.current_yaw 

        if self.docking_direction == 'backward':
            if yaw_goal_angle_error < 0:
                yaw_goal_angle_error = yaw_goal_angle_error + math.pi
            elif yaw_goal_angle_error > 0:
                yaw_goal_angle_error = yaw_goal_angle_error - math.pi
      
        return yaw_goal_angle_error

    def kalman(self, y):         
        self.P_pre = self.P + self.Q
        self.K = (self.P_pre * self.C)/((self.C*self.P_pre*self.C)+self.R)
        self.x = self.x + self.K*(y-self.C*self.x)
        self.P = (1-self.K*self.C)*self.P_pre
        return self.x

    def kalman1(self, y):  
        self.P_pre1 = self.P1 + self.Q1
        self.K1 = (self.P_pre1 * self.C1)/((self.C1*self.P_pre1*self.C1)+self.R1)
        self.x1 = self.x1 + self.K1*(y-self.C1*self.x1)
        self.P1 = (1-self.K1*self.C1)*self.P_pre1
        return self.x1

    def listener_callback(self, msg):
        self.apriltag_msg = msg
    
    # def test(self):
    def init_starting_point(self):
        try:
            if len(self.apriltag_msg.detections) == 3:

                try: # use try to prevent loss data on current time

                    # tf_robot_to_tag_left = self.tfBuffer.lookup_transform(self.base_footprint, 'tag_24', rospy.Time()) #left
                    # tf_robot_to_tag_middle = self.tfBuffer.lookup_transform(self.base_footprint, 'tag_25', rospy.Time()) #middle
                    # tf_robot_to_tag_right = self.tfBuffer.lookup_transform(self.base_footprint, 'tag_26', rospy.Time()) #right

                    tf_robot_to_tag_left = self.tfBuffer.lookup_transform(self.base_footprint, self.apriltag_left['frame_name'], rospy.Time()) #left
                    tf_robot_to_tag_middle = self.tfBuffer.lookup_transform(self.base_footprint, self.apriltag_middle['frame_name'], rospy.Time()) #middle
                    tf_robot_to_tag_right = self.tfBuffer.lookup_transform(self.base_footprint, self.apriltag_right['frame_name'], rospy.Time()) #right
                    
                    tfx1 = tf_robot_to_tag_left.transform.translation.x
                    tfy1 = tf_robot_to_tag_left.transform.translation.y
                    tfx2 = tf_robot_to_tag_middle.transform.translation.x
                    tfy2 = tf_robot_to_tag_middle.transform.translation.y
                    tfx3 = tf_robot_to_tag_right.transform.translation.x
                    tfy3 = tf_robot_to_tag_right.transform.translation.y

                    xyxy = np.array([[tfx1,tfy1], [tfx2, tfy2], [tfx3, tfy3]])

                    #find line of best fit
                    coefficients = np.polyfit(xyxy[:, 1], xyxy[:, 0], 1)
                    line_function = np.poly1d(coefficients)

                    y_min = tfy1
                    y_max = tfy3
                    x_min = line_function(y_min)
                    x_max = line_function(y_max)

                    slope = (y_max - y_min) / (x_max - x_min)              

                    slope = -1.0/slope

                    mid_point_x  = (x_max + x_min)/2.0
                    mid_point_y  = (y_max + y_min)/2.0
                
                    xx = 0.0

                    yy = (-1.0 * (mid_point_x-xx) * slope) + mid_point_y

                    yaw_robot_to_predocking = math.atan2((mid_point_y - yy), (mid_point_x - xx)) # calculate yaw of robot to pre_docking


                    tf_docking = TransformStamped()
                    tf_docking.header.stamp = rospy.Time.now()
                    tf_docking.header.frame_id = self.odom_frame
                    tf_docking.child_frame_id = 'starting_point'

                    # find tf odom to robot
                    self.tf_odom_to_robot = self.tfBuffer.lookup_transform(self.odom_frame, self.base_footprint, rospy.Time()) #middle

                    tf_docking.transform.translation.x = self.tf_odom_to_robot.transform.translation.x 
                    tf_docking.transform.translation.y = self.tf_odom_to_robot.transform.translation.y 
                    tf_docking.transform.translation.z = self.tf_odom_to_robot.transform.translation.z 
                   
                    _,_,yaw_odom_to_robot = euler_from_quaternion(self.tf_odom_to_robot.transform.rotation.x, self.tf_odom_to_robot.transform.rotation.y, self.tf_odom_to_robot.transform.rotation.z, self.tf_odom_to_robot.transform.rotation.w)
                
                    yaw_odom_to_predocking = yaw_odom_to_robot + yaw_robot_to_predocking # calculate yaw of odom to pre_docking from yaw of robot to pre_docking with yaw of odom to robot
                    
                    q = quaternion_from_euler(0.0, 0.0, yaw_odom_to_predocking) 

                    tf_docking.transform.rotation.x = q[0]
                    tf_docking.transform.rotation.y = q[1]
                    tf_docking.transform.rotation.z = q[2]
                    tf_docking.transform.rotation.w = q[3]
                    self.static_pre_docking_broadcaster.sendTransform(tf_docking)

                except:
                    pass

            else:
                pass
        
        except:
            pass
    
    def test2(self):
        try:
            
            if len(self.apriltag_msg.detections) == 3:

                try: # use try to prevent loss data on current time

                    tf_robot_to_tag_left = self.tfBuffer.lookup_transform('starting_point', self.apriltag_left['frame_name'], rospy.Time()) #left
                    tf_robot_to_tag_middle = self.tfBuffer.lookup_transform('starting_point', self.apriltag_middle['frame_name'], rospy.Time()) #middle
                    tf_robot_to_tag_right = self.tfBuffer.lookup_transform('starting_point', self.apriltag_right['frame_name'], rospy.Time()) #right
                    
                    tfx1 = tf_robot_to_tag_left.transform.translation.x
                    tfy1 = tf_robot_to_tag_left.transform.translation.y
                    tfx2 = tf_robot_to_tag_middle.transform.translation.x
                    tfy2 = tf_robot_to_tag_middle.transform.translation.y
                    tfx3 = tf_robot_to_tag_right.transform.translation.x
                    tfy3 = tf_robot_to_tag_right.transform.translation.y

                    xyxy = np.array([[tfx1,tfy1], [tfx2, tfy2], [tfx3, tfy3]])

                    #find line of best fit
                    coefficients = np.polyfit(xyxy[:, 1], xyxy[:, 0], 1)
                    line_function = np.poly1d(coefficients)

                    y_min = tfy1
                    y_max = tfy3
                    x_min = line_function(y_min)
                    x_max = line_function(y_max)

                    slope = (y_max - y_min) / (x_max - x_min)

                    slope = -1.0/slope

                    mid_point_x  = (x_max + x_min)/2.0
                    mid_point_y  = (y_max + y_min)/2.0
                    
                    self.tf_starting_point_to_robot = self.tfBuffer.lookup_transform('starting_point', self.base_footprint, rospy.Time()) 
                    self.tf_dist = self.tfBuffer.lookup_transform('starting_point', self.apriltag_middle['frame_name'], rospy.Time()) #middle


                    # find tf  tag25 to robot 
                    try:
                        tf_tag_25_to_base_footprint = self.tfBuffer.lookup_transform(self.apriltag_middle['frame_name'], self.base_footprint, rospy.Time()) #middle
                    except:
                        pass
                    else:
                        self.tf_tag25_to_robot = tf_tag_25_to_base_footprint



                    # if ( (self.first_pre_docking_distance - self.distance_goal_tolerance_first_predocking_) <= self.tf_tag25_to_robot.transform.translation.z <= (self.first_pre_docking_distance + self.distance_goal_tolerance_first_predocking_)) and abs(self.tf_tag25_to_robot.transform.translation.x)<0.10 :
                    #     self.first_point = False        
                    # if self.first_point == True and (((self.first_pre_docking_distance + self.distance_goal_tolerance_first_predocking_) < self.tf_tag25_to_robot.transform.translation.z < (self.first_pre_docking_distance - self.distance_goal_tolerance_first_predocking_)) or abs(self.tf_tag25_to_robot.transform.translation.x)>0.05 ):
                    #     xx = self.tf_dist.transform.translation.x - self.first_pre_docking_distance
                    
                    if self.first_point == True :

                        if ((self.first_pre_docking_distance + self.distance_goal_tolerance_first_predocking_) < self.tf_tag25_to_robot.transform.translation.z) or (self.tf_tag25_to_robot.transform.translation.z < (self.first_pre_docking_distance - self.distance_goal_tolerance_first_predocking_)): # Ex. 1.0 meters < dist < 0.9 meters
                           
                            xx = self.tf_dist.transform.translation.x - self.first_pre_docking_distance # fixed position of pre-docking
                            
                            
                        elif abs(self.tf_tag25_to_robot.transform.translation.x)>0.05: # error of side > x meters
                            xx = self.tf_dist.transform.translation.x - self.first_pre_docking_distance # fixed position of pre-docking
                          
                        
                        elif (self.first_pre_docking_distance + self.distance_goal_tolerance_first_predocking_) > self.tf_tag25_to_robot.transform.translation.z > (self.first_pre_docking_distance - self.distance_goal_tolerance_first_predocking_): # Ex. 1.0 meters > dist > 0.9 meters: # robot is around pre-docking position and error of side < x meters
                            if abs(self.tf_tag25_to_robot.transform.translation.x)<0.05: # error of side < x meters
                                self.first_point = False
                               
                    
                    else:
                        # palabola equation
                        # scale = (-0.05*(self.tf_tag25_to_robot.transform.translation.x-2.0)**2) + 0.2 
                        # scale = (-0.0375*(self.tf_tag25_to_robot.transform.translation.z+2.0)**2) + 0.15 
                        # scale = (-0.025*(self.tf_tag25_to_robot.transform.translation.x-2.0)**2) + 0.1
                        # xx =  self.tf_starting_point_to_robot.transform.translation.x - scale 
                        scale = (self.palabola_beta*(self.tf_tag25_to_robot.transform.translation.z + self.palabola_dist)**2) + self.palabola_alpha
                        # scale = -0.2
                        xx =  self.tf_starting_point_to_robot.transform.translation.x - scale  
                    
                         
                    
                    yy = (-1.0 * (mid_point_x-xx) * slope) + mid_point_y
                    
                    yaw = math.atan2((mid_point_y - yy), (mid_point_x - xx))

                    q = quaternion_from_euler(0,0,yaw)
                    
                    tf_docking = TransformStamped()
                    tf_docking.header.stamp = rospy.Time.now()
                    tf_docking.header.frame_id = 'starting_point'
                    tf_docking.child_frame_id = 'pre_docking'
                    # tf_docking.transform.translation.x = self.kalman(xx)
                    # tf_docking.transform.translation.y = self.kalman1(yy)
                    tf_docking.transform.translation.x = xx
                    tf_docking.transform.translation.y = yy
                    tf_docking.transform.translation.z = 0.0
                    tf_docking.transform.rotation.x = q[0]
                    tf_docking.transform.rotation.y = q[1]
                    tf_docking.transform.rotation.z = q[2]
                    tf_docking.transform.rotation.w = q[3]
                
                    self.static_pre_docking_broadcaster.sendTransform(tf_docking)                

                except:
                    pass
        

            # elif any(self.apriltag_msg.detections[i].id[0] == 25 for i in range(len(self.apriltag_msg.detections))) :
            elif any(self.apriltag_msg.detections[i].id[0] == self.apriltag_middle['tag_id'] for i in range(len(self.apriltag_msg.detections))) :
               
                # find tf  tag25 to robot 
                try:
                    tf_tag_25_to_base_footprint = self.tfBuffer.lookup_transform(self.apriltag_middle['frame_name'], self.base_footprint, rospy.Time()) #middle
                except:
                    pass
                else:
                    self.tf_tag25_to_robot = tf_tag_25_to_base_footprint

                if self.first_point == False:
                    try:
                        if abs(self.tf_tag25_to_robot.transform.translation.z) < self.min_dist_to_deatect_all_tags:
                        
                            q = quaternion_from_euler(-1.57079, 1.57079, 0.0) #rpy = pi/2, pi/2, 0.0

                            tf_docking = TransformStamped()
                            tf_docking.header.stamp = rospy.Time()
                            tf_docking.header.frame_id = self.apriltag_middle['frame_name']
                            tf_docking.child_frame_id = 'pre_docking'
                            tf_docking.transform.translation.x = 0.0
                            tf_docking.transform.translation.y = 0.0
                            tf_docking.transform.translation.z = self.docking_distance
                            tf_docking.transform.rotation.x = q[0]
                            tf_docking.transform.rotation.y = q[1]
                            tf_docking.transform.rotation.z = q[2]
                            tf_docking.transform.rotation.w = q[3]
                        
                            self.static_pre_docking_broadcaster.sendTransform(tf_docking)
                    except:
                        pass
            else:
                pass
        
        except:
            pass
    
    def test3(self):
        cmd_vel_msg = geometry_msgs.msg.Twist()
        try:
            robot2predocking = self.tfBuffer.lookup_transform(self.odom_frame, 'pre_docking', rospy.Time())
            robot2predocking_2 = self.tfBuffer.lookup_transform(self.base_footprint, 'pre_docking', rospy.Time())

            #calculate direction
            if robot2predocking_2.transform.translation.x >= 0.0 :
                self.current_direction = 1.0
            elif robot2predocking_2.transform.translation.x < 0.0 :
                self.current_direction = -1.0
            
            # calculate desired goal
            self.desired_x = round(robot2predocking.transform.translation.x,3)
            self.desired_y = round(robot2predocking.transform.translation.y,3)
            _, _, self.desired_theta = euler_from_quaternion(robot2predocking.transform.rotation.x , robot2predocking.transform.rotation.y, robot2predocking.transform.rotation.z, robot2predocking.transform.rotation.w)
            self.desired_theta = round(self.desired_theta,3)
            
            #cal error (linear and angular)
            distance_to_goal = self.get_distance_to_goal()
            heading_error = self.get_heading_error()
            yaw_goal_error = self.get_radians_to_goal()
        
            #tolerance depend on distance to pre-docking
            if self.first_point == True:
                if (self.first_pre_docking_distance - self.distance_goal_tolerance_first_predocking_) >= abs(self.tf_tag25_to_robot.transform.translation.z) or abs(self.tf_tag25_to_robot.transform.translation.z) >= (self.first_pre_docking_distance + self.distance_goal_tolerance_first_predocking_) :
                    self.distance_goal_tolerance = self.distance_goal_tolerance_first_predocking_ 

                elif abs(self.tf_tag25_to_robot.transform.translation.z) >= self.distance_goal_tolerance_first_predocking_ :
                    self.distance_goal_tolerance = self.distance_goal_tolerance_first_predocking_ 
            else:
                self.distance_goal_tolerance = self.distance_goal_tolerance_next_predocking_
            
            # PID
            if abs(self.tf_tag25_to_robot.transform.translation.z) > self.docking_distance + self.distance_goal_tolerance_last_predocking :
                if (math.fabs(distance_to_goal) > self.distance_goal_tolerance and self.reached_distance_goal == False ):
                    if abs(self.tf_tag25_to_robot.transform.translation.z) >= self.yaw_adjust_disable_by_distance_x:
                        if (math.fabs(heading_error) > self.heading_error_threshold): # ถ้า heading error มีค่ามากเกินกว่า self.heading_error_threshold ให้ปรับแต่ heading 
                            cmd_vel_msg.angular.z = self.kp_angular * heading_error
                        else:
                            cmd_vel_msg.linear.x = self.current_direction * self.kp_linear * distance_to_goal
                            cmd_vel_msg.angular.z = self.kp_angular * heading_error
                            
                    else: # ถ้าระยะจากจากหุ่นยนต์ไปที่ middle tag น้อยกว่า self.yaw_adjust_disable_by_distance_x จะไม่ปรับ heading
                        cmd_vel_msg.linear.x = self.current_direction * self.kp_linear * distance_to_goal

                # Orient towards the yaw goal angle 
                elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance) and (abs(self.tf_tag25_to_robot.transform.translation.z) > self.yaw_adjust_disable_by_distance_x): #0.45 # ถ้าระยะมากกว่า x จะปรับ heading ตอนไปถึง goal  
                    cmd_vel_msg.angular.z = (self.kp_angular) * yaw_goal_error
                    self.reached_distance_goal = True
                
                # Goal achieved, go to the next goal  
                else:
                    # Go to the next goal
                    cmd_vel_msg = Twist()
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.angular.z = 0.0
                    self.cmd_vel.publish(cmd_vel_msg)

                    self.reached_distance_goal = False
                    self.first_point = False

                    # 'Arrived at perpendicular line. Going straight to AprilTag...'
                    self.reached_distance_goal = False   
            else:
                try:
                    report_error = self.tfBuffer.lookup_transform('camera_link', 'tag_25', rospy.Time()) 
                    _,_,yaw_report_error = euler_from_quaternion(report_error.transform.rotation.x, report_error.transform.rotation.y, report_error.transform.rotation.z, report_error.transform.rotation.w)
                    yaw_report_error = abs(yaw_report_error) - math.pi/2.0
                    yaw_report_error = round(math.degrees(yaw_report_error),2)
                    print("report error -> dist_y_error = {} meters / angle_error = {} degrees".format(round(report_error.transform.translation.y,3), yaw_report_error))

                    self.docking_state = 'succeed'
                    print(self.docking_state)
                except:
                    print("waitting for report...")
        except:
            pass  

        if cmd_vel_msg.linear.x > self.linear_velocity_max:
            cmd_vel_msg.linear.x = self.linear_velocity_max
        elif cmd_vel_msg.linear.x < -self.linear_velocity_max:
            cmd_vel_msg.linear.x = -self.linear_velocity_max

        if cmd_vel_msg.angular.z > self.angular_velocity_max:
            cmd_vel_msg.angular.z = self.angular_velocity_max
        elif cmd_vel_msg.angular.z < -self.angular_velocity_max:
            cmd_vel_msg.angular.z = -self.angular_velocity_max
            
        self.cmd_vel.publish(cmd_vel_msg)

    
    def demo_callback(self, timer):
        #####################################################################   
        ########################### docking state ###########################
        ##################################################################### 

        if self.docking_state == 'init':
            self.time_ = 0.0
            self.reached_distance_goal = False 
            
            self.docking_state = 'waitting'
        
        elif self.docking_state == 'waitting':
            self.init_starting_point()
            try:
                tf_robot_to_tag_left = self.tfBuffer.lookup_transform('starting_point', self.apriltag_left['frame_name'], rospy.Time()) #left
                tf_robot_to_tag_middle = self.tfBuffer.lookup_transform('starting_point', self.apriltag_middle['frame_name'], rospy.Time()) #middle
                tf_robot_to_tag_right = self.tfBuffer.lookup_transform('starting_point', self.apriltag_right['frame_name'], rospy.Time()) #right
            except:
                pass
            else:
                self.docking_state = 'waitting2'

        elif self.docking_state == 'waitting2':
            self.time_ = self.time_ + self.duration_time
            self.test2()
            if self.time_ >= self.delay_before_start: #delay by user
                self.docking_state = 'going to the docking'
        
        elif self.docking_state == 'going to the docking':
            self.test2()
            self.test3()
            
        # print(self.docking_state)
        
if __name__ == '__main__':

    print("entering main")
    rospy.init_node('custom_talkerr')
    try:
        DemoNode()
        print("entering Try")
        rospy.spin()
    except rospy.ROSInterruptException:
        print("exception thrown")
        pass