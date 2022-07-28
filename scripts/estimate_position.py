#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np

from trajectory_predictor.utils.SplineOptimizer import SplineOptimizer
from trajectory_predictor.model.MeanPredictor.MeanPredictor import MeanPredictor
from trajectory_predictor.dataset.Dataset import Dataset

class Agent(object):
    def __init__(self, spline, predictor_model, max_queue_length=10):
        self.max_queue_length = max_queue_length
        self.spline = spline
        self.predictor_model = predictor_model
        self.dataset = Dataset()
        self.dataset.optimizer = spline
        self.traj_pub = rospy.Publisher('/prediction_marker', Marker, queue_size=1)
        self.pf_sub = rospy.Subscriber('/pf/pose/odom', Odometry, self.pf_loc_callback, queue_size=1)
        self.trajectory_history = []
        self.r = rospy.Rate(10)

    def pf_loc_callback(self, scan_msg):
        position = scan_msg.pose.pose.position
        position = np.array([position.x, position.y])
        curvilinear_pos = self.spline.euclidean_to_curvilinear(position)
        self.add_to_history(curvilinear_pos)
        # TODO history probably reversed
        # print(self.trajectory_history)
        if len(self.trajectory_history) == self.max_queue_length:
            prediction = self.predict()
            self.publish_prediction(prediction)
            print("Predicting...")
            self.r.sleep()

    def publish_prediction(self, trajectory):
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

    def predict(self):
        history = np.vstack(self.trajectory_history)
        series = self.dataset.history_to_series(history)
        series = series[:, :-1] # removing curvatures which aren't used
        prediction = self.predictor_model.predict(series, 300)
    
        # Converting delta progress to progress in predicion
        # TODO this can be danggerous if not done correctly
        prediction[:, 0] = np.cumsum(prediction[:, 0]) + history[-1, 0]

        for i in range(len(prediction)):
            # TODO move this to the spline (curvilinear to euclidean)
            current = self.spline.interp(prediction[i, 0])
            bef = self.spline.interp(prediction[i, 0]-0.001)
            tangencial= current-bef
            tangencial_normalized = tangencial/np.linalg.norm(tangencial)
            normal = np.array([tangencial_normalized[1], -tangencial_normalized[0]])
            delta = prediction[i, 1]*normal #vector from centerline to point
            prediction[i] = self.spline.interp(prediction[i, 0])+delta
        
        return prediction


if __name__ == '__main__':
    track = np.loadtxt(f'../../gmapping/new_map.csv', delimiter=',')
    rospy.init_node('progress_printer')

    # Initializing track centerline
    optim = SplineOptimizer(track)
    optim.sample_spline_by_tolerance(0.1, optimize=False, verbose=False)

    # Initializing predictor
    model = MeanPredictor()
    
    progress_printer = Agent(optim, model)
    rospy.spin()
