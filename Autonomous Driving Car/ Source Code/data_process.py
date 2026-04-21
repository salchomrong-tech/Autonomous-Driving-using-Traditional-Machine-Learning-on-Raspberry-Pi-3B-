
import numpy as np
import cv2

class Horizontal:
    def __init__(self):
        self.image = None
        self.height = 480
        self.width = 640
        self.line = 20
        self.list_line = np.array([pixel for pixel in range(self.height-40, self.height//2, -self.line)])
        self.kernel = np.array([-1, -1, -1, 0, 1, 1, 1])
        self.filter = np.array([-1, 0, 1])
        self.perfect_data = None
        self.boundary_data = np.zeros((4, len(self.list_line)))  # 0:y, 1:x1, 2:x2, 3:x

    def process_data(self, frame):
        data_deploy = np.zeros((20))
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        index = 0
        for line in self.list_line:
            right_data = self.image[line, self.width//2:]
            left_data = self.image[line, self.width//2-1::-1]

            right_raw = np.convolve(right_data, self.kernel, mode='valid')
            left_raw = np.convolve(left_data, self.kernel, mode='valid')

            right_bound = np.where(abs(right_raw) > 80)[0]
            left_bound = np.where(abs(left_raw) > 80)[0]

            self.boundary_data[0, index] = line
            if right_bound.size != 0 and left_bound.size != 0:
                self.boundary_data[1, index] = 320-left_bound[0]
                self.boundary_data[2, index] = 320+right_bound[0]
            elif right_bound.size != 0:
                self.boundary_data[1, index] = 0
                self.boundary_data[2, index] = 320+right_bound[0]
            elif left_bound.size != 0:
                self.boundary_data[1, index] = 320-left_bound[0]
                self.boundary_data[2, index] = 640
            index += 1

        self.boundary_data[3, :] = (self.boundary_data[1, :] + self.boundary_data[2, :]) // 2

        conv_mid = np.convolve(self.boundary_data[3, :], self.filter, mode='valid')
        valid_filter = np.where(abs(conv_mid) > 75)[0]

        if valid_filter.size > 0:
            cut_index = valid_filter[0]
            data_deploy[:cut_index] = self.boundary_data[1, 0:cut_index]
            data_deploy[10:cut_index+10] = self.boundary_data[2, 0:cut_index]
        else:
            data_deploy[0:10] = self.boundary_data[1, 0:10]
            data_deploy[10:20] = self.boundary_data[2, 0:10]

        return data_deploy
