
import numpy as np
import math
class angle:
    def __init__(self):
        self.midpoint_self = np.zeros((3))
        self.midpoint_right = np.zeros((3))
        self.y = 60
        self.x = 0
        self.a = 90
        self.p = 0
        self.turn_left = 0
        self.turn_right = 0

    def range_angle(self, data, predict):
        self.midpoint_left = data[0:3][data[0:3]!=0]
        self.midpoint_right = data[10:13][data[10:13]!=0]
        if len(self.midpoint_left) > len(self.midpoint_right):
            self.midpoint_left=self.midpoint_left[0:len(self.midpoint_right)]
        elif len(self.midpoint_left) < len(self.midpoint_right):
            self.midpoint_right=self.midpoint_right[0:len(self.midpoint_left)]
        if predict == 2:
            x = (self.midpoint_left+self.midpoint_right)/2-30
        elif predict == 3:
            x = (self.midpoint_left+self.midpoint_right)/2+30
        else:
            x = (self.midpoint_left+self.midpoint_right)/2
        x_p = np.mean(x)
        if x_p < 320:
            self.x = 320-np.mean(x)
        else:
            self.x = np.mean(x)-320
        self.a = math.degrees(math.atan(self.x/self.y))

        if x_p < 320:
            self.a = 90+self.a
        else:
            self.a = 90-self.a

        if self.a > 150:
            self.a = 150
        elif self.a < 30:
            self.a = 30
        return self.a

    def calculate_angle(self, data, predict):
        if self.turn_left != 0:
            if predict == 3 or predict == 5:
                self.range_angle(data, predict)
                self.turn_left = 0
            else:
                return self.a
        elif self.turn_right != 0:
            if predict == 2 or predict == 4:
                self.range_angle(data, predict)
                self.turn_right = 0
            else:
                return self.a
        else:
            self.range_angle(data, predict)
        return self.a
