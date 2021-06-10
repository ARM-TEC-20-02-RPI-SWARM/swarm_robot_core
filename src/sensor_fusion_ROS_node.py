#!/usr/bin/env python
import numpy as np
import rospy
import threading

import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import tf_conversions # quaternion stuff

from std_msgs.msg import String # For UWB messages
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D

# Local python files
from multilateration import *
from UWB_odom_kalman_filter import *
from uwb_parsing import *

#TF_FRAME_NAME_UWB = "base_link_UWB"
#TF_FRAME_NAME_FUSED = "base_link_RPI_fusion"
TF_FRAME_NAME_UWB = "OARBOT_1_base_link_UWB"
TF_FRAME_NAME_FUSED = "OARBOT_1_base_link_RPI_fusion"
POSITION_FEEDBACK_TOPIC_NAME = "oarbot_1_position"

# XY location of tags on robot
# (Code assume tags are at same Z postion)
# Convention is X axis points to front of robot

# RIDGEBACK
# TAG_LOC_FRONT = np.array([[0.5054],[0.0254]])
# TAG_LOC_BACK = np.array([[-0.17405],[0.0254]])

# OARBOT 1
TAG_LOC_FRONT = 0.0254*np.array([[9.5],[-4.5]])
TAG_LOC_BACK  = 0.0254*np.array([[0.5],[4.5]])

# Max time (in seconds) to consider UWB readings to happen simulataneously
# Note that UWB readings happen at 10 Hz
UWB_TIMEOUT = 0.03

# timestamp of reading
front_t = 0.
back_t = 0.

# Matrix of anchor positions
front_anchors = 0.
back_anchors = 0.

# Vector of distances to each anchor
front_dists = 0.
back_dists = 0.

# Kalman filter state, covariance, and time
state = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
cov   = 100.0**2 * np.eye(6)
kalman_time = 0.

kalman_lock = threading.Lock()


def uwb_serial_front_callback(data):
	#rospy.loginfo("front uwb callback" + str(rospy.Time.now().to_sec()))
	global front_t, front_anchors, front_dists

	#valid, anchor_mat, dists = parse_lec_line(str(data).split('"')[1])
	valid, anchor_mat, dists = parse_lec_line(data.data)
	if not valid:
		return

	# front_t = data.header.stamp
	front_t = rospy.Time.now().to_sec()
	#rospy.loginfo("front_t:" + str(front_t))
	front_anchors = anchor_mat
	front_dists = dists

	#rospy.loginfo((front_t - back_t).to_sec())
	if (front_t - back_t) < UWB_TIMEOUT:
		combine_uwb_readings()

def uwb_serial_back_callback(data):
	#rospy.loginfo("back uwb callback" + str(rospy.Time.now().to_sec()))
	global back_t, back_anchors, back_dists

	valid, anchor_mat, dists = parse_lec_line(str(data).split('"')[1])
	if not valid:
		rospy.loginfo("NOT VALID")
		rospy.loginfo(type(data))
		rospy.loginfo(str(data))
		return

	# back_t = data.header.stamp
	back_t = rospy.Time.now().to_sec();
	#rospy.loginfo("back_t: " + str(back_t))
	back_anchors = anchor_mat
	back_dists = dists

	#rospy.loginfo((back_t - front_t).to_sec())
	if (back_t - front_t) < UWB_TIMEOUT:
		combine_uwb_readings()

def combine_uwb_readings():
	#rospy.loginfo("combine_uwb_readings")
	global state, cov, kalman_time

	#rospy.loginfo("front_t - back_t:" + str(front_t-back_t))
	#rospy.loginfo("back_t:" + str(back_t))
	#rospy.loginfo("kalman_time:" + str(kalman_time))

	

	# Kalman filter
	kalman_lock.acquire()
	dt = max(front_t, back_t) - kalman_time;
	if dt < 0:	
		kalman_lock.release()
		rospy.logwarn("Dropping UWB reading | dt = " + str(dt))
		return
	if dt > 1:
		rospy.logwarn("Limiting UWB timestep to 1 | dt = " + str(dt))
		dt = 1
	kalman_time =  max(front_t, back_t);

	# Multilateration
	uwb_pos, rmse = tag_pair_min_z(front_anchors, back_anchors,
	front_dists, back_dists, TAG_LOC_FRONT, TAG_LOC_BACK)

	#print("uwb_pos")
	#print(uwb_pos[[0,1,3],np.newaxis])
	state, cov, kalman_pos = EKF_UWB(state, cov, dt, uwb_pos[[0,1,3],np.newaxis], rmse)
	#state, cov, kalman_pos = EKF_UWB(state, cov, dt, uwb_pos[[0,1,3],np.newaxis], 0.01)
	kalman_lock.release()


	# Change XYT to XYZT
	# Let Z = 0

	kalman_pos_xyzt = np.array([[0.0],[0.0],[0.0],[0.0]])
	kalman_pos_xyzt[0] = kalman_pos[0]
	kalman_pos_xyzt[1] = kalman_pos[1]
	kalman_pos_xyzt[3] = kalman_pos[2]
	# Publish
	#kalman_loc_pub.publish(
	#	xyzt2TF(kalman_pos, kalman_time, "map"))
	#uwb_loc_pub.publish(xyzt2TF(uwb_pos, kalman_time, "map"))

	br = tf2_ros.TransformBroadcaster()
	t = xyzt2TF(kalman_pos_xyzt, rospy.Time.from_sec(kalman_time), "map", TF_FRAME_NAME_FUSED)
	br.sendTransform(t)

	t = xyzt2TF(uwb_pos, rospy.Time.from_sec(kalman_time), "map", TF_FRAME_NAME_UWB)
	br.sendTransform(t)
	publish_position()
	#print(rmse*1e2)
	#rospy.loginfo(str(np.diag(cov)))

def odom_callback(data):
	#rospy.loginfo("Odom callback")
	global state, cov, kalman_time

	# Assemble the measurement vector
	x_d = data.twist.twist.linear.x
	y_d = data.twist.twist.linear.y
	theta_d = data.twist.twist.angular.z
	meas = np.array([[x_d],[y_d],[theta_d]])


	# Kalman filter
	kalman_lock.acquire()

	#t = data.header.stamp.to_sec()
	t = rospy.Time.now().to_sec()
	dt = (t - kalman_time)
	if dt < 0:	
		kalman_lock.release()
		rospy.logwarn("Dropping odom reading | dt = " + str(dt))
		return
	if dt > 1:
		rospy.logwarn("Limiting odom timestep to 1 | dt = " + str(dt))
		dt = 1
	
	kalman_time = t
	state, cov, kalman_pos = EKF_odom(state, cov, dt, meas)
	kalman_lock.release()

	# Change XYT to XYZT
	# Let Z = 0
	kalman_pos = np.block([[kalman_pos[0:1+1]],[0.],[kalman_pos[2]]])

	# Publish
	br = tf2_ros.TransformBroadcaster()
	t = xyzt2TF(kalman_pos, rospy.Time.from_sec(kalman_time), "map", TF_FRAME_NAME_FUSED);
	br.sendTransform(t)
	publish_position()
	#kalman_loc_pub.publish(
	#	xyzt2TF(kalman_pos, kalman_time, "map"))

def oarbot_vel_cmd_callback(data):
	#rospy.loginfo("oarbot_vel_cmd_callback")
	global state, cov, kalman_time

	# Assemble the measurement vector
	x_d = data.linear.x
	y_d = data.linear.y
	theta_d = data.angular.z
	meas = np.array([[x_d],[y_d],[theta_d]])


	# Kalman filter
	kalman_lock.acquire()

	#t = data.header.stamp.to_sec()
	t = rospy.Time.now().to_sec()
	dt = (t - kalman_time)
	if dt < 0:	
		kalman_lock.release()
		rospy.logwarn("Dropping odom reading | dt = " + str(dt))
		return
	if dt > 1:
		rospy.logwarn("Limiting odom timestep to 1 | dt = " + str(dt))
		dt = 1
	
	kalman_time = t
	state, cov, kalman_pos = EKF_odom(state, cov, dt, meas)
	kalman_lock.release()

	# Change XYT to XYZT
	# Let Z = 0
	kalman_pos_xyzt = np.array([[0.0],[0.0],[0.0],[0.0]])
	kalman_pos_xyzt[0] = kalman_pos[0]
	kalman_pos_xyzt[1] = kalman_pos[1]
	kalman_pos_xyzt[3] = kalman_pos[2]


	# Publish
	br = tf2_ros.TransformBroadcaster()
	t = xyzt2TF(kalman_pos_xyzt, rospy.Time.from_sec(kalman_time), "map", TF_FRAME_NAME_FUSED);
	br.sendTransform(t)
	publish_position()
	#kalman_loc_pub.publish(
	#	xyzt2TF(kalman_pos, kalman_time, "map"))

def xyzt2TF(xyzt, ros_time, header_frame_id, child_frame_id):
	xyzt = xyzt.flatten()
	'''
	Converts a numpy vector [x; y; z; theta]
	into a tf2_msgs.msg.TFMessage message
	'''
	t = geometry_msgs.msg.TransformStamped()

	t.header.frame_id = header_frame_id
	#t.header.stamp = ros_time #rospy.Time.now()
	t.header.stamp = rospy.Time.now()
	t.child_frame_id = child_frame_id
	t.transform.translation.x = xyzt[0]
	t.transform.translation.y = xyzt[1]
	t.transform.translation.z = xyzt[2]

	q = tf_conversions.transformations.quaternion_from_euler(0, 0,xyzt[3])
	t.transform.rotation.x = q[0]
	t.transform.rotation.y = q[1]
	t.transform.rotation.z = q[2]
	t.transform.rotation.w = q[3]

	return t

def publish_position():
	pos_msg = Pose2D()
	pos_msg.x = state[0][0]
	pos_msg.y = state[1][0]
	pos_msg.theta = state[2][0]
	pos_pub.publish(pos_msg)

if __name__ == '__main__':
	rospy.init_node('sensor_fusion', anonymous=True)
	kalman_time = rospy.Time.now().to_sec()

	# Subscribe to UWB tags
	print("starting uwb subscribers")
	#rospy.Subscriber("uwb_serial_front", String, uwb_serial_front_callback)
	#rospy.Subscriber("uwb_serial_back", String, uwb_serial_back_callback)
	rospy.Subscriber("oarbot/uwb_serial_front", String, uwb_serial_front_callback)
	rospy.Subscriber("oarbot/uwb_serial_back", String, uwb_serial_back_callback)

	# Subscribe to odometry
	# print("starting odom subscribers")
	# rospy.Subscriber(
	# 	"ridgeback_velocity_controller/odom", Odometry, odom_callback)

	# Subscribe to OARbot vel_cmd
	print("starting OARbot vel_cmd subscribers")
	rospy.Subscriber(
		"vel_feedback", Twist, oarbot_vel_cmd_callback)

	# Publish position
	pos_pub = rospy.Publisher(POSITION_FEEDBACK_TOPIC_NAME, Pose2D, queue_size=10)

	rospy.spin()