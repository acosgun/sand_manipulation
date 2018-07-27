import numpy as np
import rospy

from std_msgs.msg import Int32MultiArray

def timer_callback(event):
    points = Int32MultiArray()
    points.data = [360,350,250,350]
    pub.publish(points)

pub = None

if __name__ == '__main__':
    rospy.init_node('fake_push_publisher', anonymous=True)
    pub = rospy.Publisher('points', Int32MultiArray, queue_size=10)
    rospy.Timer(rospy.Duration(0.5), timer_callback)
    rospy.spin()
