#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Indrajeet yadav'
__version__ = '0.1'
__license__ = 'Nil'


import rospy
import time
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
#from tf import TransformListener, TransformerROS
#from nav_msgs.msg import Odometry
#from sensor_msgs.msg import Imu
from pyquaternion import Quaternion 
#from parameters import parameters
#from isy_geometric_controller.msg import Desired_Trajectory, control_inputs
from gazebo_msgs.msg import LinkStates
#import message_filters
from numpy import array as ar # just to reduce typing 
from numpy import transpose as tp # just to reduce typing 
from numpy import multiply as mult # just to reduce typing 
from tf import transformations
#from trajectory import tajectory 
#from mav_msgs.msg import Actuators
#from isy_geometric_controller.msg import velocities_from_navigation_function
#from scipy.interpolate import splrep, splev

class move_turtlebot(object): 
    def __init__(self):
        self.counter = 0
        self.pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size = 1)
        #self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 1)
        try:
            odom = rospy.Subscriber('/gazebo/link_states', LinkStates, self.callback, tcp_nodelay = True)
            #rospy.spin()
        except:
            print('problem subscribing to one of the topic above')
   
    def callback(self, odom): 
        Msg = Twist()
        #Msg = ModelState()
        if self.counter == 0: 
            #X = ar([[odom.pose[1].position.x], [odom.pose[1].position.y], [odom.pose[1].position.z]])
            q = Quaternion(odom.pose[1].orientation.w, odom.pose[1].orientation.x, odom.pose[1].orientation.y, odom.pose[1].orientation.z)       
            print q
            #q_inv = q.inverse
            #print q_inv
            euler = transformations.euler_from_quaternion([q[1], q[2], q[3], q[0]], axes = 'sxyz')
            print euler
        else: 

            Msg.linear.x = 1; Msg.linear.y = 0.0; Msg.linear.z = 0.0
            Msg.angular.x = 0.0; Msg.angular.y = 0.0; Msg.angular.z = 0.0265

        self.pub.publish(Msg)
        self.counter = self.counter+1

if __name__ == '__main__':
    name = 'firefly'
    rospy.init_node('move_turtlebot', anonymous=False, log_level=rospy.DEBUG)
    r = rospy.Rate(100)

    try: 
        while not rospy.is_shutdown(): 
            c = move_turtlebot()
            rospy.spin()
    except rospy.ROSInterruptException(): 
        pass

