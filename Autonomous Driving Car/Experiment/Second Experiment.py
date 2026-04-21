from data_process import Horizontal
from flask import Flask, Response
from picamera2 import Picamera2
from angle_data import angle
import threading
import queue
import numpy as np
import serial
import joblib
import time
import cv2
import os
model = joblib.load(r"/home/admin/received_frames/SVM.pkl")
app = Flask(__name__)
# Replace multiprocessing.Queue with queue.Queue
frame_queue = queue.Queue(maxsize=2)
class Mainprocess:
    def __init__(self):
        self.line = 20
        self.height = 480
        self.width = 640
        self.ang = 90
        self.ang_before = 90
        self.actions = ['Forward', 'Turn right', 'Turn left', 'Turn right', 'Turn left']
        self.list_line = np.array(
            [pixel for pixel in range(self.height - 40, self.height // 2, -self.line)]
        )
        self.ser = None
        self.picam2 = None
        self.h = Horizontal()
        self.a = angle()
        self.SAVE_DIR = "/home/admin/received_frames/streamed_images"
        self.number_image = 0
        self.time_list = np.zeros((1200, 5))
        # Thread control
        self.stop_thread = False
        self.serial_thread = None
    # ---------------- Arduino thread -----------------
    def read_arduino(self):
        if self.ser is None:
            return
        while not self.stop_thread:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line.isdigit() and int(line) == 1:
                        if self.number_image < len(self.time_list):
                            self.time_list[self.number_image, 4] = 1
            except serial.SerialException:
                print("[ERROR] Arduino disconnected")
                break
            time.sleep(0.001)
    # ---------------- Hardware init -----------------
    def init_hardware(self):
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            time.sleep(2)
        except serial.SerialException:
            print("Error: Arduino not connected!")
            self.ser = None
        if self.ser:
            self.serial_thread = threading.Thread(target=self.read_arduino, daemon=True)
            self.serial_thread.start()
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        time.sleep(1)
    # ---------------- Processing -----------------
    def marking(self, frame):
        if self.number_image >= 1000:
            csv_path = os.path.join(self.SAVE_DIR, "time_list.csv")
            np.savetxt(csv_path, self.time_list, delimiter=",", fmt="%.3f",
                       header="process_data_ms,total_processing_ms,full_loop_ms", comments="")
            print(f"[INFO] time_list saved")
        t1 = time.time()
        data = self.h.process_data(frame)
        t2 = time.time()
        if self.number_image < len(self.time_list):
            self.time_list[self.number_image, 0] = (t2 - t1) * 1000
        if np.all(data == 0):
            self.ang = self.ang_before
            if self.number_image < len(self.time_list):
                self.time_list[self.number_image, 4] = 1
        else:
            predict = int(model.predict([data])[0])
            self.ang = self.a.calculate_angle(data, predict)
            ang_int = max(0, min(255, int(round(self.ang))))
            if self.ser:
                self.ser.write(bytes([ang_int]))
            x_r = data[10:20][data[10:20] != 0]
            x_l = data[0:10][data[0:10] != 0]
            n_points = min(len(x_r), len(x_l))
            x_r = x_r[:n_points]
            x_l = x_l[:n_points]
            x = (x_r + x_l) // 2
            y = self.list_line[:n_points]
        t3 = time.time()
        if self.number_image < len(self.time_list):
            self.time_list[self.number_image, 1] = (t3 - t2) * 1000
        if not np.all(data == 0):
            for i in range(n_points):
                cv2.circle(frame, (int(x[i]), int(y[i])), 3, (0, 0, 255), -1)
                cv2.circle(frame, (int(x_l[i]), int(y[i])), 3, (255, 0, 0), -1)
                cv2.circle(frame, (int(x_r[i]), int(y[i])), 3, (255, 0, 0), -1)
            cv2.putText(frame, f"SVM: {self.actions[predict - 1]}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.ang_before = self.ang
        t4 = time.time()
        if self.number_image < len(self.time_list):
            self.time_list[self.number_image, 2] = (t4 - t3) * 1000
        self.number_image += 1
        return frame
    # ---------------- Main loop -----------------
    def processing_loop(self):
        self.init_hardware()
        while True:
            start = time.time()
            frame = self.picam2.capture_array("main")
            frame = self.marking(frame)
            if not frame_queue.full():
                frame_queue.put(frame)
            stop = time.time()
            if self.number_image < len(self.time_list):
                self.time_list[self.number_image - 1, 3] = (stop - start) * 1000
    def stop(self):
        self.stop_thread = True
        if self.serial_thread:
            self.serial_thread.join()
# ---------------- Flask Streaming ----------------
def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() +
                   b'\r\n')
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ---------------- Thread Wrappers ----------------
def start_processing(handler):
    handler.processing_loop()
def start_streaming():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
# ---------------- MAIN ----------------
if __name__ == "__main__":
    handler = Mainprocess()
    t1 = threading.Thread(target=start_processing, args=(handler,))
    t2 = threading.Thread(target=start_streaming)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
