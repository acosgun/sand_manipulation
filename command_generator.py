#!/usr/bin/env python

import rospy
from std_msgs.msg import String

if __name__ == '__main__':
    rospy.init_node('command_generator', anonymous=True)
    pub = rospy.Publisher('/commands', String, queue_size=10)
    print("Keys:")
    print("a + Enter: Execute ANN action recommendation")
    print("b + Enter: Execute Poly Regression action recommendation")
    print("c + Enter: Execute Contour Center action recommendation")
    print("d + Enter: Execute Max Contour Difference action recommendation")
    print("h + Enter: Robot goes to home position. This is buggy, though. Not recommended.")
    print("q + Enter: Kill command_generator")

    while True:
        key_input = raw_input("---Command: ")
        if key_input == "q":
            exit(0)
        str_msg = String()
        str_msg.data = key_input
        pub.publish(str_msg)
