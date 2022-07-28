#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np

from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer

class Agent(object):
    def __init__(self, spline, max_queue_length=300):
        self.max_queue_length = max_queue_length
        self.spline = spline
        self.traj_pub = rospy.Publisher('/prediction_marker', Marker, queue_size=1)
        self.pf_sub = rospy.Subscriber('/pf/pose/odom', Odometry, self.pf_loc_callback)
        self.trajectory_history = []

    def pf_loc_callback(self, scan_msg):
        position = scan_msg.pose.pose.position
        position = np.array([position.x, position.y])
        curvilinear_pos = self.spline.euclidean_to_curvilinear(position)
        self.add_to_history(position)

        self.publish_predicion(self.trajectory_history)
        print(position, curvilinear_pos)

    def publish_predicion(self, trajectory):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.LINE_STRIP
        marker.action = marker.ADD

        # Marker scale
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03

        # Marker color
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        # Marker orientaiton
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Marker position
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        # Marker line points
        marker.points = []
        for p in trajectory:
            point = Point()
            point.x = p[0]
            point.y = p[1]
            marker.points.append(point)

        # Publish the Marker
        self.traj_pub.publish(marker)

    def add_to_history(self, position):
        if len(self.trajectory_history) >= self.max_queue_length:
            del self.trajectory_history[0]
        self.trajectory_history.append(position)


if __name__ == '__main__':
    track = np.loadtxt(f'../../gmapping/new_map.csv', delimiter=',')
    optim = SplineOptimizer(track)
    optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)
    rospy.init_node('progress_printer')
    progress_printer = Agent(optim)
    rospy.spin()
