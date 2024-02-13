#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import geometry_msgs.msg

from std_msgs.msg import String
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray

import tf
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
    
        self.cmd_vel = rospy.Publisher('/coconut_vel',geometry_msgs.msg.Twist, queue_size=10)
        self.subscription = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.listener_callback)
        self.subscription  # prevent unused variable warning
        self.apriltag_msg = None
        self.odom_sub = rospy.Subscriber('/odom_gz', Odometry, self.odom_callback)
        self.odom_current = PoseStamped()
        self.current_yaw = None
        
        print("creating a publisher")
        self.duration_time = 0.05
        self.timer = rospy.Timer(rospy.Duration(self.duration_time), self.demo_callback)
        print("timer called")

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.static_pre_docking_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.tf_robot_to_tag25 = None
        self.tf_odom_to_robot = None

        self.K = 0
        self.x = 0
        self.P = 0
        self.P_pre = 0
        self.R = 0.00000015523403536447
        self.C = 1
        self.Q = 0.00009

        self.K1 = 0
        self.x1 = 0
        self.P1 = 0
        self.P_pre1 = 0
        self.R1 = 0.00008339191740914060
        self.C1 = 1
        self.Q1 = 0.00009

        self.docking_state = 'init'
        self.time_ = 0.0

        self.desired_x = None
        self.desired_y = None
        self.desired_theta = None

        # Declare angle metrics in radians
        self.heading_tolerance = 0.01
        self.yaw_goal_tolerance = 0.005

        # Declare linear and angular velocities
        self.linear_velocity_max = 0.09  # meters per second
        self.angular_velocity_max = 0.5 # radians per second

        # Declare distance metrics in meters
        self.distance_goal_tolerance = 0.01
        self.reached_distance_goal = False  

        self.kp = 1.2
        self.kp_linear = 0.4

        self.docking_distance = 0.38 #from apriltag to center of robot

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
        heading_error = desired_heading - self.current_yaw

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
    
    def demo_callback(self, timer):
        # declare
        cmd_vel_msg = geometry_msgs.msg.Twist()
        try:
            if len(self.apriltag_msg.detections) == 3:
                print('pass1')

                try: # use try to prevent loss data on current time
                    tf_odom_to_tag24 = self.tfBuffer.lookup_transform('odom_gz', 'tag_24', rospy.Time()) #left
                    tf_odom_to_tag25 = self.tfBuffer.lookup_transform('odom_gz', 'tag_25', rospy.Time()) #middle
                    tf_odom_to_tag26 = self.tfBuffer.lookup_transform('odom_gz', 'tag_26', rospy.Time()) #right
                    
                    tfx1 = tf_odom_to_tag24.transform.translation.x
                    tfy1 = tf_odom_to_tag24.transform.translation.y
                    tfx2 = tf_odom_to_tag25.transform.translation.x
                    tfy2 = tf_odom_to_tag25.transform.translation.y
                    tfx3 = tf_odom_to_tag26.transform.translation.x
                    tfy3 = tf_odom_to_tag26.transform.translation.y

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

                    # find tf robot to tag25
                    try:
                        tf_base_footprint_to_tag25 = self.tfBuffer.lookup_transform('base_footprint', 'tag_25', rospy.Time()) #middle
                    except:
                        pass
                    else:
                        self.tf_robot_to_tag25 = tf_base_footprint_to_tag25
                    
                    # find tf odom to robot
                    self.tf_odom_to_robot = self.tfBuffer.lookup_transform('odom_gz', 'base_footprint', rospy.Time()) #middle



                    if self.tf_robot_to_tag25.transform.translation.x > 1.0:
                                xx = self.tf_odom_to_robot.transform.translation.x + self.tf_robot_to_tag25.transform.translation.x - 0.9 # next point = 0.9 m from station
                    else:
                        # palabola equation
                        scale = (-0.05*(self.tf_robot_to_tag25.transform.translation.x-2.0)**2) + 0.2
                        # scale = (-0.0525*(tf_odom2station.transform.translation.x-2.0)**2) + 0.21 
                        xx = self.tf_odom_to_robot.transform.translation.x + scale
                    
                    yy = (-1.0 * (mid_point_x-xx) * slope) + mid_point_y

                    yaw = math.atan2((mid_point_y - yy), (mid_point_x - xx))

                    q = quaternion_from_euler(0,0,yaw)

                    tf_docking = TransformStamped()
                    tf_docking.header.stamp = rospy.Time()
                    tf_docking.header.frame_id = 'odom_gz'
                    tf_docking.child_frame_id = 'pre_docking'
                    tf_docking.transform.translation.x = self.kalman(xx)
                    tf_docking.transform.translation.y = self.kalman1(yy)
                    tf_docking.transform.translation.z = 0.0
                    tf_docking.transform.rotation.x = q[0]
                    tf_docking.transform.rotation.y = q[1]
                    tf_docking.transform.rotation.z = q[2]
                    tf_docking.transform.rotation.w = q[3]
                
                    self.static_pre_docking_broadcaster.sendTransform(tf_docking)
                except:
                    pass
        

            elif any(self.apriltag_msg.detections[i].id[0] == 25 for i in range(len(self.apriltag_msg.detections))):
                print('pass2')

                # find tf robot to tag25
                try:
                    tf_base_footprint_to_tag25 = self.tfBuffer.lookup_transform('base_footprint', 'tag_25', rospy.Time()) #middle
                except:
                    pass
                else:
                    self.tf_robot_to_tag25 = tf_base_footprint_to_tag25

                try:
                    if self.tf_robot_to_tag25.transform.translation.x < 0.5:
                    
                        q = quaternion_from_euler(-1.57079, 1.57079, 0.0) #rpy = pi/2, pi/2, 0.0

                        tf_docking = TransformStamped()
                        tf_docking.header.stamp = rospy.Time()
                        tf_docking.header.frame_id = 'tag_25'
                        tf_docking.child_frame_id = 'pre_docking'
                        tf_docking.transform.translation.x = 0.0
                        tf_docking.transform.translation.y = 0.0
                        # tf_docking.transform.translation.z = ss.transform.translation.x
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
    

        #####################################################################   
        ########################### docking state ###########################
        ##################################################################### 

        if self.docking_state == 'init':
            self.time_ = 0.0
            self.docking_state = 'waitting'

        elif self.docking_state == 'waitting':
            self.time_ = self.time_ + self.duration_time
            if self.time_ >= 5.0: # wait for 5 sec before moving
                self.docking_state = 'going to the docking'
        
        elif self.docking_state == 'going to the docking':
            try:
                robot2predocking = self.tfBuffer.lookup_transform('odom_gz', 'pre_docking', rospy.Time())

                # defined x,y for caliculate theta
                self.desired_x = round(robot2predocking.transform.translation.x,3)
                self.desired_y = round(robot2predocking.transform.translation.y,3)

                # self.get_logger().info('distance_to_goal: "%s"' % self.desired_y)

                _, _, self.desired_theta = euler_from_quaternion(robot2predocking.transform.rotation.x , robot2predocking.transform.rotation.y, robot2predocking.transform.rotation.z, robot2predocking.transform.rotation.w)
                self.desired_theta = round(self.desired_theta,3)

                distance_to_goal = self.get_distance_to_goal()
                heading_error = self.get_heading_error()
                yaw_goal_error = self.get_radians_to_goal()

                if (math.fabs(distance_to_goal) > self.distance_goal_tolerance and self.reached_distance_goal == False ):

                    # if (math.fabs(heading_error) > 0.2) and self.tf_robot_to_tag25.transform.translation.x > 0.5: # ถ้าระยะน้อยกว่า x จะไม่ปรับ major heading 
                    if (math.fabs(heading_error) > 0.2) and self.tf_robot_to_tag25.transform.translation.x > 0.6: # ถ้าระยะน้อยกว่า x จะไม่ปรับ major heading 
                        print("heading")
                    # if (math.fabs(heading_error) > 0.5) :
                        # self.get_logger().info('PID state: "%s"' % '0')
                    
                        # if heading_error > 0:
                        #     cmd_vel_msg.angular.z = self.angular_velocity
                        # else:
                        #     cmd_vel_msg.angular.z = -self.angular_velocity

                        cmd_vel_msg.angular.z = self.kp * heading_error

                        if cmd_vel_msg.angular.z > self.angular_velocity_max:
                            cmd_vel_msg.angular.z = self.angular_velocity_max
                        elif cmd_vel_msg.angular.z < -self.angular_velocity_max:
                            cmd_vel_msg.angular.z = -self.angular_velocity_max
                    
                    else: # heading_error = 0.3 and dist err > 0
                        # print("go")
                        # print(self.kp_linear * distance_to_goal)
                        # self.get_logger().info('PID state: "%s"' % '1')
                        cmd_vel_msg.linear.x = self.kp_linear * distance_to_goal
                        cmd_vel_msg.angular.z = self.kp * heading_error
                        # if (self.desired_y - self.odom_current.pose.position.y < 0.01) and self.tf_robot_to_tag25.transform.translation.x < 0.6: # ถ้าระยะน้อยกว่า x จะไม่ปรับ heading
                        if (self.desired_y - self.odom_current.pose.position.y < 0.01) and self.tf_robot_to_tag25.transform.translation.x < 0.55: # ถ้าระยะน้อยกว่า x จะไม่ปรับ heading
                            print("heading2")
                            cmd_vel_msg.angular.z = 0.0

                        if cmd_vel_msg.angular.z > self.angular_velocity:
                            cmd_vel_msg.angular.z = self.angular_velocity
                        elif cmd_vel_msg.angular.z < -self.angular_velocity:
                            cmd_vel_msg.angular.z = -self.angular_velocity
                        
                        
                        if cmd_vel_msg.linear.x > self.linear_velocity_max:
                            cmd_vel_msg.linear.x = self.linear_velocity_max
                
                # Orient towards the yaw goal angle
                elif (math.fabs(yaw_goal_error) > self.yaw_goal_tolerance) and self.tf_robot_to_tag25.transform.translation.x > 0.5 : # ถ้าระยะมากกว่า x จะปรับ heading ตอนไปถึง goal 
                    
                    # self.get_logger().info('PID state: "%s"' % '2')
                    # self.get_logger().info('yaw error: "%s"' % math.fabs(yaw_goal_error))
                    
                    # if yaw_goal_error > 0:
                    #     cmd_vel_msg.angular.z = self.angular_velocity
                    # else:
                    #     cmd_vel_msg.angular.z = -self.angular_velocity

                    cmd_vel_msg.angular.z = (self.kp-0.3) * yaw_goal_error

                    if (self.desired_y - self.odom_current.pose.position.y < 0.008) and self.tf_robot_to_tag25.transform.translation.x < 0.58: # ถ้าระยะ y คลาดเคลื่อนน้อยมาก และ ระยะที่เหลือน้อยกว่า x+8cm จะไม่ปรับ heading ตอนไปถึง goal
                    # if (self.desired_y - self.odom_current.pose.position.y < 0.008) and tf_odom2station.transform.translation.x < 0.63:
                            
                            cmd_vel_msg.angular.z = 0.0

                
                    
                    self.reached_distance_goal = True
                
                # # Goal achieved, go to the next goal  
                else:
                    # Go to the next goal
                    cmd_vel_msg = Twist()
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.angular.z = 0.0
                    self.cmd_vel.publish(cmd_vel_msg)

                #     self.get_logger().info('Arrived at perpendicular line. Going straight to ARTag...')
                    self.reached_distance_goal = False   

                    if self.tf_robot_to_tag25.transform.translation.x < self.docking_distance + self.distance_goal_tolerance : # 0.4 + tolerance
                        self.docking_state = 'succeed'

            except:
                pass   

        # cmd_vel_msg.angular.z = 0.1
        # cmd_vel_msg.linear.x = 0.0
        
        #if backward:
        # cmd_vel_msg.linear.x = -1.0 * cmd_vel_msg.linear.x
        # cmd_vel_msg.angular.z = -1.0 * cmd_vel_msg.angular.z 
            

        self.cmd_vel.publish(cmd_vel_msg)

if __name__ == '__main__':

    print("entering main")
    rospy.init_node('custom_talker')
    try:
        DemoNode()
        print("entering Try")
        rospy.spin()
    except rospy.ROSInterruptException:
        print("exception thrown")
        pass