#!/usr/bin/env python3

'''
This python file runs a ROS-node of name drone_control which holds the position of Swift-Drone on the given dummy.
This node publishes and subsribes the following topics:

    PUBLICATIONS			SUBSCRIPTIONS
        /drone_command			/whycon/poses
        /alt_error				/pid_tuning_altitude
        /pitch_error			/pid_tuning_pitch
        /roll_error				/pid_tuning_roll

Rather than using different variables, use list. eg : self.setpoint = [1,2,3], where index corresponds to x,y,z ...rather than defining self.x_setpoint = 1, self.y_setpoint = 2
CODE MODULARITY AND TECHNIQUES MENTIONED LIKE THIS WILL HELP YOU GAINING MORE MARKS WHILE CODE EVALUATION.
'''

# Importing the required libraries
from luminosity_drone.msg import Biolocation
from imutils import contours
from skimage import measure
import numpy as np
import imutils
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv

from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time
import math

class Swift():
    """docstring for swift"""
    def __init__(self):
        rospy.init_node('drone_control') # initializing ros node with name drone_control

        # This corresponds to your current position of drone. This value must be updated each time in your whycon callback
        # [x,y,z]
        self.drone_position = [0.0,0.0,0.0]

        # [x_setpoint, y_setpoint, z_setpoint]
        self.setpoints = [
            [0, 0, 24.68],
            [8.3, 0, 24.68],
            [8.3, 8.3, 24.68],
            [0, 8.3, 24.68],
            [-8.3, 8.3, 24.68],
            [-8.3, 0, 24.68],
            [-8.3, -8.3, 24.68],
            [0, -8.3, 24.68],
            [8.3, -8.3, 24.68],

            # [8.3, -8.3, 37],
            # [11, 11, 37]
        ]

        self.error = [0, 0, 0]

        #Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = swift_msgs()
        self.cmd.rcRoll = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcThrottle = 1500
        self.cmd.rcAUX1 = 1500
        self.cmd.rcAUX2 = 1500
        self.cmd.rcAUX3 = 1500
        self.cmd.rcAUX4 = 1500

        #initial setting of Kp, Kd and ki for [roll, pitch, throttle]. eg: self.Kp[2] corresponds to Kp value in throttle axis
        #after tuning and computing corresponding PID parameters, change the parameters
        self.Kp = [130 * 0.06, 285 * 0.06, 700 * 0.06]
        self.Ki = [2 * 0.0004, 4 * 0.0004, 512 * 0.0004]
        self.Kd = [240 * 0.3, 480 * 0.3, 1100 * 0.3]
        self.Kpid = 1.0
        self.Kpid_z = 1.0

        self.error = [0, 0, 0]
        self.prev_error = [0, 0, 0]
        self.error_sum = [0, 0, 0]

        self.max_values = [2000, 2000, 2000]
        self.min_values = [1000, 1000, 1000]

        # # This is the sample time in which you need to run pid. Choose any time which you seem fit. Remember the stimulation step time is 50 ms
        # self.sample_time = 0.050 # in seconds

        # Publishing /drone_command, /alt_error, /pitch_error, /roll_error
        self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
        self.alt_error_pub = rospy.Publisher('/alt_error', Float64, queue_size=1)
        self.pitch_error_pub  = rospy.Publisher('/pitch_error', Float64, queue_size=1)
        self.roll_error_pub = rospy.Publisher('/roll_error', Float64, queue_size=1)

        # Subscribing to /whycon/poses, /pid_tuning_altitude, /pid_tuning_pitch, pid_tuning_roll
        rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
        rospy.Subscriber('/pid_tuning_altitude',PidTune,self.altitude_set_pid)
        rospy.Subscriber('/pid_tuning_pitch',PidTune,self.pitch_set_pid)
        rospy.Subscriber('/pid_tuning_roll',PidTune,self.roll_set_pid)

        self.arm() # ARMING THE DRONE


    # Disarming condition of the drone
    def disarm(self):
        self.cmd.rcAUX4 = 1100
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)

    # Arming condition of the drone : Best practise is to disarm and then arm the drone.
    def arm(self):
        self.disarm()

        self.cmd.rcRoll = 1500
        self.cmd.rcYaw = 1500
        self.cmd.rcPitch = 1500
        self.cmd.rcThrottle = 1000
        self.cmd.rcAUX4 = 1500
        self.command_pub.publish(self.cmd)	# Publishing /drone_command
        rospy.sleep(1)

    # Whycon callback function
    # The function gets executed each time when /whycon node publishes /whycon/poses
    def whycon_callback(self,msg):
        self.drone_position[0] = msg.poses[0].position.x
        self.drone_position[1] = msg.poses[0].position.y
        self.drone_position[2] = msg.poses[0].position.z

    # Callback function for /pid_tuning_altitude
    # This function gets executed each time when /tune_pid publishes /pid_tuning_altitude
    def altitude_set_pid(self,alt):
        self.Kp[2] = alt.Kp * 0.06 # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[2] = alt.Ki * 0.0004
        self.Kd[2] = alt.Kd * 0.3

    def pitch_set_pid(self,pitch):
        self.Kp[1] = pitch.Kp * 0.06 # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[1] = pitch.Ki * 0.0004
        self.Kd[1] = pitch.Kd * 0.3

    def roll_set_pid(self,roll):
        self.Kp[0] = roll.Kp * 0.06 # This is just for an example. You can change the ratio/fraction value accordingly
        self.Ki[0] = roll.Ki * 0.0004
        self.Kd[0] = roll.Kd * 0.3

    def clamp(self, value, index):
        return int(max(min(value, self.max_values[index]), self.min_values[index]))

    def pid(self):
        # Steps:
        # 	1. Compute error in each axis. eg: error[0] = self.drone_position[0] - self.setpoint[0] ,where error[0] corresponds to error in x...
        #	2. Compute the error (for proportional), change in error (for derivative) and sum of errors (for integral) in each axis. Refer "Understanding PID.pdf" to understand PID equation.
        #	3. Calculate the pid output required for each axis. For eg: calcuate self.out_roll, self.out_pitch, etc.
        #	4. Reduce or add this computed output value on the avg value ie 1500. For eg: self.cmd.rcRoll = 1500 + self.out_roll. LOOK OUT FOR SIGN (+ or -). EXPERIMENT AND FIND THE CORRECT SIGN
        #	5. Don't run the pid continously. Run the pid only at the a sample time. self.sampletime defined above is for this purpose. THIS IS VERY IMPORTANT.
        #	6. Limit the output value and the final command value between the maximum(2000) and minimum(1000)range before publishing. For eg : if self.cmd.rcPitch > self.max_values[1]:
        #																														self.cmd.rcPitch = self.max_values[1]
        #	7. Update previous errors.eg: self.prev_error[1] = error[1] where index 1 corresponds to that of pitch (eg)
        #	8. Add error_sum

        p = [0, 0, 0]
        i = [0, 0, 0]
        d = [0, 0, 0]
        # self.error = [0, 0, 0]
        for indx in range(3):
            # change: error is now calculated in the main loop.
            # self.error[indx] = self.drone_position[indx] - self.setpoints[setpoint_index][indx]

            p[indx] = self.error[indx] * self.Kp[indx]

            self.error_sum[indx] += self.error[indx]
            i[indx] = self.error_sum[indx] * self.Ki[indx]

            d[indx] = (self.error[indx] - self.prev_error[indx]) * self.Kd[indx]
            self.prev_error[indx] = self.error[indx]

        self.cmd.rcRoll = 1500 - (p[0] + i[0] + d[0]) * self.Kpid
        self.cmd.rcPitch = 1500 + (p[1] + i[1] + d[1]) * self.Kpid
        self.cmd.rcThrottle = 1540 + (p[2] + i[2] + d[2]) * self.Kpid_z

        self.cmd.rcRoll = self.clamp(self.cmd.rcRoll, 0)
        self.cmd.rcPitch = self.clamp(self.cmd.rcPitch, 1)
        self.cmd.rcThrottle = self.clamp(self.cmd.rcThrottle, 2)

        self.roll_error_pub.publish(self.error[0])
        self.pitch_error_pub.publish(self.error[1])
        self.alt_error_pub.publish(self.error[2])
        self.command_pub.publish(self.cmd)


class Imag():
    def __init__(self):
        rospy.Subscriber("/swift/camera_rgb/image_raw", Image, self.callback)
        self.shape = (0, 0)
        self.coords = []
        self.centroid = [0.0, 0.0]  # [x, y] coords of the centroid of the alien

    def callback(self, data):
        br = CvBridge()
        # rospy.loginfo("receiving video frame")
        try:
            current_frame = br.imgmsg_to_cv2(data, "passthrough")
            self.shape = current_frame.shape
        except CvBridgeError as e:
            rospy.logerr(f"Interface Error: {e}")
        current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        current_frame = cv.blur(current_frame, (10, 10))
        threshold_return,current_frame = cv.threshold(current_frame, 128, 255, cv.THRESH_BINARY)
        kernel = np.ones((10, 10), np.uint8)
        current_frame = cv.erode(current_frame, kernel, iterations = 1)
        current_frame = cv.dilate(current_frame, kernel, iterations = 1)
        (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(current_frame, 4, cv.CV_32S)
        self.coords = self.func1(current_frame, numLabels, labels)
        # if len(self.coords) > 0:
        #     self.compute_centroid()
        #     rospy.loginfo('computed centroid: ' + str(self.centroid))

        # rospy.loginfo(self.shape)
        # rospy.loginfo(self.coords)
        cv.imshow("camera", current_frame)
        cv.waitKey(2)

    def compute_centroid(self):
        x = 0
        y = 0
        for led in self.coords:
            x += led[0]
            y += led[1]
        x /= len(self.coords)
        y /= len(self.coords)

        self.centroid[0] = x
        self.centroid[1] = y

    def func1(self, current_frame, numLabels, labels):
        contour_list = []
        cx, cy = 0, 0
        for i in range(1, numLabels):
            componentMask = (labels == i).astype("uint8") * 255
            contours, hierarchy = cv.findContours(componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour_list.append(contours)
        l_local = []
        for i in contour_list:
            cnt = i[0]
            M = cv.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            l_local.append(self.cord_translate(self.shape, (cx, cy)))
        return l_local

    def cord_translate(self, shape, led_cx_cy_ar): # note that led_cx_cy_ar is a list containing 1 set of cords and area 
        a = shape                                  # a=(img_height,img_length)
        b = (int(a[1]/2), int(a[0]/2))
        tr_px = (led_cx_cy_ar[0] - b[0], b[1] - led_cx_cy_ar[1])
        scale = (a[1] / (8.6), a[0] / (8.6))  # 8.6  as drone_cam size/grid size is 8.6 by 8.6 #note that by 2 as 0,0 is new centre,so endpt is 4,4 not 8,8
        tr_why = (tr_px[0] / scale[0], tr_px[1] / scale[1])
        return tr_why

if __name__ == '__main__':
    pub = rospy.Publisher('astrobiolocation',Biolocation,queue_size=10)
    swift_drone = Swift()
    image = Imag()
    r = rospy.Rate(20)
    msg = Biolocation()

    on_centroid = False
    done = False
    target = [11.0, 11.0, 37.0]
    step = 1.0

    setpoint_index = 0
    while not rospy.is_shutdown():
        if done:
            swift_drone.disarm()
            break
        elif on_centroid:
            for indx in range(3):
                swift_drone.error[indx] = swift_drone.drone_position[indx] - (target[indx] - 3.0 * step)
            if (math.fabs(swift_drone.error[0]) < 0.2
                and math.fabs(swift_drone.error[1]) < 0.2
                and math.fabs(swift_drone.error[2]) < 0.2):
                if step == 0.0:
                    done = True
                step = 0.0
        else:
            for indx in range(3):
                swift_drone.error[indx] = swift_drone.drone_position[indx] - swift_drone.setpoints[setpoint_index][indx]

            if len(image.coords) > 0 and swift_drone.drone_position[2] < 32:
                image.compute_centroid()
                rospy.loginfo('computed centroid: ' + str(image.centroid))
                # set the centroid coords to the error
                swift_drone.Kpid = 0.3
                swift_drone.error[0] = -image.centroid[0]
                swift_drone.error[1] = image.centroid[1]

                if (math.fabs(swift_drone.error[0]) < 0.3 and math.fabs(swift_drone.error[1] < 0.3)):
                    on_centroid = True
                    if on_centroid:
                        org_type = len(image.coords)
                        type_list = ['alien_a','alien_b','alien_c']
                        msg.organism_type = type_list[org_type-2]
                        msg.whycon_x = swift_drone.drone_position[0]
                        msg.whycon_y = swift_drone.drone_position[1]
                        msg.whycon_z = swift_drone.drone_position[2]
                        pub.publish(msg)
            elif   (math.fabs(swift_drone.error[0]) < 0.5
                and math.fabs(swift_drone.error[1]) < 0.5
                and math.fabs(swift_drone.error[2]) < 0.5):
                # swift_drone.error_sum = [0.0, 0.0, 0.0]
                setpoint_index += 1
                if setpoint_index >= len(swift_drone.setpoints):
                    rospy.loginfo('Alien not found :(')
                    break
        swift_drone.pid()
        r.sleep()
