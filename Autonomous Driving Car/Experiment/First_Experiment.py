from data_process import Horizontal
from picamera2 import Picamera2
from angle_data import angle
import threading
import numpy as np
import serial
import joblib
import time
import cv2
import os

# ---------------- Load model ----------------
model = joblib.load("/home/admin/received_frames/SVM.pkl")
class Mainprocess:
    def __init__(self):
        self.height = 480
        self.width = 640
        self.line = 20
        self.ang = 90
        self.ang_before = 90
        self.actions = ['Forward', 'Turn right', 'Turn left', 'Turn right', 'Turn left']
        self.list_line = np.array(
            [pixel for pixel in range(self.height - 40, self.height // 2, -self.line)]
        )
        self.picam2 = None
        self.ser = None
        self.h = Horizontal()
        self.a = angle()
        self.SAVE_DIR = "/home/admin/received_frames/streamed_images"
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        self.number_image = 0
        self.MAX_FRAMES = 1000
        self.time_list = np.zeros((self.MAX_FRAMES, 5))
        # Arduino thread
        self.stop_thread = False
        self.serial_thread = None
        self.arduino_signal = 0
    # ---------------- Arduino thread ----------------
    def read_arduino(self):
        while not self.stop_thread:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode().strip()
                    if line.isdigit():
                        self.arduino_signal = int(line)
            except:
                break

    # ---------------- Hardware initialization ----------------
    def init_hardware(self):
        try: # Arduino
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            time.sleep(2)
            self.serial_thread = threading.Thread(
                target=self.read_arduino, daemon=True
            )
            self.serial_thread.start()
        except:
            print("[WARN] Arduino not connected")
        # Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)

    # ---------------- Frame processing ----------------
    def process_frame(self, frame):
        idx = self.number_image
        t0 = time.time()
        data = self.h.process_data(frame)
        t1 = time.time()
        self.time_list[idx, 0] = (t1 - t0) * 1000

        if np.all(data == 0):
            self.ang = self.ang_before
            self.time_list[idx, 4] = 1
        else:
            predict = int(model.predict([data])[0])
            self.ang = self.a.calculate_angle(data, predict)
            ang_int = int(np.clip(self.ang, 0, 255))
            if self.ser:
                self.ser.write(bytes([ang_int]))
            x_r = data[10:20][data[10:20] != 0]
            x_l = data[0:10][data[0:10] != 0]
            n = min(len(x_r), len(x_l))
            x = (x_r[:n] + x_l[:n]) // 2
            y = self.list_line[:n]
            for i in range(n):
                cv2.circle(frame, (int(x[i]), int(y[i])), 3, (0, 0, 255), -1)
        t2 = time.time()
        self.time_list[idx, 1] = (t2 - t1) * 1000
        cv2.putText(frame, f"Angle: {self.ang:.2f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        filename = f"{self.SAVE_DIR}/frame_{idx}.jpg" # Save image
        cv2.imwrite(filename, frame)
        t3 = time.time()
        self.time_list[idx, 2] = (t3 - t2) * 1000
        self.time_list[idx, 3] = (t3 - t0) * 1000
        self.ang_before = self.ang
        self.number_image += 1
    # ---------------- Main loop ----------------
    def run(self):
        self.init_hardware()
        try:
            while self.number_image < self.MAX_FRAMES:
                frame = self.picam2.capture_array("main")
                self.process_frame(frame)
        finally:
            self.shutdown()
    # ---------------- Cleanup ----------------
    def shutdown(self): # data into a csv file
        self.stop_thread = True
        if self.serial_thread:
            self.serial_thread.join()
        csv_path = f"{self.SAVE_DIR}/time_list.csv"
        np.savetxt(
            csv_path,
            self.time_list,
            delimiter=",",
            fmt="%.3f",
            header="process_data,angle_calc,draw,full_loop,arduino_signal",
            comments=""
        )
        print("[INFO] Saved:", csv_path)
# ---------------- Run ----------------
if __name__ == "__main__":
    Mainprocess().run()
