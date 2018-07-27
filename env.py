#!/usr/bin/env python

import os
import sys
from gym import spaces

import rospy
import rosbag
import cv2
import cv_bridge
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf.transformations as tft
from position_control import *

import numpy as np

from kinova_msgs.msg import PoseVelocity
from kinova_msgs.msg import JointAngles
import kinova_msgs.msg
import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg
import sensor_msgs.msg
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

class SortEnv:
    def __init__(self, NL, OPL, TL):
        self.time_limit        = TL
        self.num_locations     = NL
        self.obj_per_loc       = OPL
        self.num_objects       = self.num_locations * self.obj_per_loc
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.num_locations + self.num_objects*self.num_locations,))
        self.action_space      = spaces.Discrete(self.num_objects + self.num_locations)
        self.remove_penalty    = -0.05
        self.placement_reward  =  0.025
        self.bin_reward        =  0.15
        self.bin_penalty       = -0.3
        self.complete_reward   =  1.0
        self.class_colors = [(0,0,255), (255,0,0), (0,255,0), (0,64,128)]
        self.slow = False
        self.reset()

    #--------------------------------------------------------------------------

    def reset(self):

        self.timestep = 0
        self.grasped  = None
        self.objects  = np.random.randint(0, self.num_locations, (self.num_objects,))

        home = [241.6, 165.8, 52.665, 385.23, 392.93, -239.86, 0.0]
        move_to_pos(home)

        while self._done(): return self.reset()
        return self._obs()

    #--------------------------------------------------------------------------

    def step(self, act):
        score = self.num_correct() - self.num_incorrect()

        if act < self.num_objects and self.grasped is None:
            self.grasped = act

        if act >= self.num_objects and self.grasped is not None: # drop
            self.objects[self.grasped] = act-self.num_objects
            self.grasped = None


        reward = float((self.num_correct() - self.num_incorrect()) - score) / self.num_objects

        self.timestep += 1
        return self._obs(), reward + int(self._all_correct()), self._done(), self._info()

    #--------------------------------------------------------------------------

    def num_correct(self):
        return sum([int(self._target_location(object_id) == location and self.grasped != object_id) for object_id,location in enumerate(self.objects)])

    def num_incorrect(self):
        return sum([int(self._target_location(object_id) != location and self.grasped != object_id) for object_id,location in enumerate(self.objects)])

    def _bins_solved(self):
        bins_solved = 0
        for b in range(self.num_locations):
            bin_solved = True
            for o in range(b*self.obj_per_loc,(b+1)*self.obj_per_loc):
                if self._target_location(o) != self.objects[o]: bin_solved = False
            if bin_solved: bins_solved += 1
        return bins_solved

    #--------------------------------------------------------------------------

    def _target_location(self, object_id):
        return object_id // self.obj_per_loc

    #--------------------------------------------------------------------------

    def _obs(self):
        obs = np.zeros(self.observation_space.shape, np.float32)
        if self.grasped is not None: obs[self._target_location(self.grasped)] = 1
        for i in range(self.num_objects): obs[self.num_locations+i*self.num_locations+self.objects[i]] = 1
        return obs

    #--------------------------------------------------------------------------

    def _all_correct(self):
        return all([(self._target_location(object_id) == location) for object_id,location in enumerate(self.objects)])

    #--------------------------------------------------------------------------

    def _done(self):
        return (self.timestep >= self.time_limit or self._all_correct())

    #--------------------------------------------------------------------------

    def _info(self):
        pass



    

