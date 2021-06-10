#!/usr/bin/env python
import numpy as np
import rospy

from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import Joy

import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import tf_conversions # quaternion stuff

from velocity_controller import *

POSITION_FEEDBACK_TOPIC_NAME = "/oarbot_2/position"
FRAME_NAME = "oarbot_2_qd"
VEL_CMD_INPUT_TOPIC_NAME = "spacenav/twist/repub"
CONTROL_CMD_PUBLISH_TOPIC_NAME = '/spacenav/twist/repub3'

# POSITION_FEEDBACK_TOPIC_NAME = "ridgeback_position"
# FRAME_NAME = "ridgeback_qd"
# VEL_CMD_INPUT_TOPIC_NAME = "spacenav/twist/repub"
# CONTROL_CMD_PUBLISH_TOPIC_NAME = 'cmd_vel'

#TIMESTEP = 0.01

VEL_LIMIT = np.array([[2],[2],[2.]])

# Integrated desired position
q_desired = state = np.array([[0.0],[0.0],[0.0]])
prev_integration_time = 0

output_enable = False

state_pos = np.array([[0.0],[0.0],[0.0]])

def desired_vel_callback(data):
	global q_desired, prev_integration_time

	current_time = rospy.Time.now().to_sec()
	dt = current_time - prev_integration_time
	prev_integration_time = current_time
	if(dt > 1):
		dt = 0

	# Data input is q_desired_dot
	q_desired_dot = np.array([[0.0],[0.0],[0.0]])
	q_desired_dot[0] = data.linear.x
	q_desired_dot[1] = data.linear.y
	q_desired_dot[2] = data.angular.z



	# Update q_desired with integration
	q_desired = q_desired + dt * q_desired_dot
	q_desired[2] = wrapToPi(q_desired[2])

	# Calculate commanded velocity
 	desired_state = np.block([[q_desired],[q_desired_dot]])
	cmd_vel = control_law(desired_state, state_pos, VEL_LIMIT)

	# Publish commanded velocity
	if(output_enable):
		cmd_vel_msg = Twist()
		cmd_vel_msg.linear.x = cmd_vel[0][0]
		cmd_vel_msg.linear.y = cmd_vel[1][0]
		cmd_vel_msg.angular.z = cmd_vel[2][0]
		vel_cmd_pub.publish(cmd_vel_msg)
	else:
		cmd_vel_msg = Twist()
		cmd_vel_msg.linear.x = 0
		cmd_vel_msg.linear.y = 0
		cmd_vel_msg.angular.z = 0
		vel_cmd_pub.publish(cmd_vel_msg)

	# Publish TF frame
	br = tf2_ros.TransformBroadcaster()
	t = xyt2TF(q_desired, "map", FRAME_NAME);
	br.sendTransform(t)

def space_mouse_button_callback(data):
	global output_enable
	output_enable = data.buttons[0]

def state_feedback_callback(data):
	global state_pos
	state_pos[0][0] = data.x
	state_pos[1][0] = data.y
	state_pos[2][0] = data.theta

def xyt2TF(xyt, header_frame_id, child_frame_id):
	xyt = xyt.flatten()
	'''
	Converts a numpy vector [x; y; z; theta]
	into a tf2_msgs.msg.TFMessage message
	'''
	t = geometry_msgs.msg.TransformStamped()

	t.header.frame_id = header_frame_id
	#t.header.stamp = ros_time #rospy.Time.now()
	t.header.stamp = rospy.Time.now()
	t.child_frame_id = child_frame_id
	t.transform.translation.x = xyt[0]
	t.transform.translation.y = xyt[1]
	t.transform.translation.z = 0

	q = tf_conversions.transformations.quaternion_from_euler(0, 0,xyt[2])
	t.transform.rotation.x = q[0]
	t.transform.rotation.y = q[1]
	t.transform.rotation.z = q[2]
	t.transform.rotation.w = q[3]

	return t


def wrapToPi(a):
	'''
	Wraps angle to [-pi,pi)
	'''
	return ((a+np.pi) % (2*np.pi))-np.pi

if __name__ == '__main__':
	rospy.init_node('closed_loop_velocity_controller', anonymous=True)
	prev_integration_time = rospy.Time.now().to_sec()

	# Subscribe to space mouse velocity commands
	rospy.Subscriber(VEL_CMD_INPUT_TOPIC_NAME, Twist, desired_vel_callback)

	# Subscribe to space mouse buttons
	rospy.Subscriber("spacenav/joy", Joy, space_mouse_button_callback)

	# Subscribe to Kalman Filter position
	rospy.Subscriber(POSITION_FEEDBACK_TOPIC_NAME, Pose2D,state_feedback_callback)

	# Publish command velocity
	vel_cmd_pub = rospy.Publisher(CONTROL_CMD_PUBLISH_TOPIC_NAME, Twist, queue_size=10)

	rospy.spin()