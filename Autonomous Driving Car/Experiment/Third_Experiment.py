from data_process import Horizontal
from flask import Flask, Response
from picamera2 import Picamera2
from angle_data import angle
import multiprocessing
import threading
import numpy as np
import serial
import joblib
import time
import cv2
import os

model = joblib.load(r"/home/admin/received_frames/SVM.pkl")

app = Flask(__name__)
frame_queue = multiprocessing.Queue(maxsize=2)

class Mainprocess:
    def __init__(self):
        self.line = 20
        self.height = 480
        self.width = 640
        self.ang = 90
        self.ang_before = 90
        self.actions = ['Forward', 'Turn right', 'Turn left', 'Turn right', 'Turn left']
        self.list_line = np.array([pixel for pixel in range(self.height - 40, self.height // 2, -self.line)])
        self.ser = None
        self.picam2 = None
        self.h = Horizontal()
        self.a = angle()
        self.SAVE_DIR = "/home/admin/received_frames/streamed_images"
        self.number_image = 0   # save 1 image every 5 frames
        self.time_list = np.zeros((1200,5))

        # Arduino threading
        self.arduino_data = 50  # default value
        self.stop_thread = False
        self.serial_thread = None

    # ---------------- Arduino thread -----------------
    def read_arduino(self):
        """Continuously read data from Arduino"""
        if self.ser is None:
            return
        while not self.stop_thread:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8').strip()
                    if int(line) == 1:
                        self.time_list[self.number_image, 4] = 1
            except serial.SerialException:
                print("[ERROR] Arduino disconnected")
                break
    # -------------------------------------------------

    def init_hardware(self):
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            time.sleep(2)
        except serial.SerialException:
            print("Error: Arduino not connected!")
            self.ser = None

        # start Arduino reading thread
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

    def marking(self, frame):
        # Save time_list when reaching limit
        if self.number_image >= 1000:
            csv_path = os.path.join(self.SAVE_DIR, "time_list.csv")
            os.makedirs(self.SAVE_DIR, exist_ok=True)
            np.savetxt(csv_path, self.time_list, delimiter=",", fmt="%.3f",
                       header="process_data_ms,total_processing_ms,full_loop_ms", comments="")
            print(f"[INFO] time_list saved to {csv_path}")
            return frame  # stop saving but continue streaming

        t1 = time.time()
        data = self.h.process_data(frame)
        t2 = time.time()
        self.time_list[self.number_image,0] = (t2-t1)*1000

        if np.all(data == 0):
            self.ang = self.ang_before
            self.time_list[self.number_image, 4] = 1
        else:
            predict = int(model.predict([data])[0])
            # You can optionally combine SVM angle with Arduino data
            self.ang = self.a.calculate_angle(data, predict)
            # Example: smooth with Arduino

            ang_int = int(round(self.ang))
            ang_int = max(0, min(255, ang_int))
            if self.ser is not None:
                self.ser.write(bytes([ang_int]))

            x_r = data[10:20][data[10:20] != 0]
            x_l = data[0:10][data[0:10] != 0]
            n_points = min(len(x_r), len(x_l))
            x_r = x_r[:n_points]
            x_l = x_l[:n_points]
            x = (x_r + x_l) // 2
            y = self.list_line[:n_points]

        t3 = time.time()
        self.time_list[self.number_image,1] = (t3-t2)*1000

        if not np.all(data == 0):
            for i in range(n_points):
                cv2.circle(frame, (int(x[i]), int(y[i])), 3, (0, 0, 255), -1)
                cv2.circle(frame, (int(x_l[i]), int(y[i])), 3, (255, 0, 0), -1)
                cv2.circle(frame, (int(x_r[i]), int(y[i])), 3, (255, 0, 0), -1)

            cv2.putText(frame, f"SVM: {self.actions[predict - 1]}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f" {self.number_image} Angle: {self.ang:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save image
        # filename = os.path.join(self.SAVE_DIR, f"frame_{self.number_image}.jpg")
        # ok, buffer = cv2.imencode(".jpg", frame)
        # if ok:
        #     with open(filename, "wb") as f:
        #         f.write(buffer.tobytes())

        self.ang_before = self.ang
        t4 = time.time()
        self.time_list[self.number_image,2] = (t4-t3)*1000
        self.number_image += 1

        return frame

    def processing_loop(self):
        os.sched_setaffinity(0, {1, 2, 3})
        print(f"[Processing] running on cores: {os.sched_getaffinity(0)}")

        self.init_hardware()

        while True:
            start = time.time()
            frame = self.picam2.capture_array("main")
            frame = self.marking(frame)
            if not frame_queue.full():
                frame_queue.put(frame)
            stop = time.time()
            self.time_list[self.number_image,3] = (stop-start)*1000
            time.sleep(max(0,0.05-(stop-start)))

    def stop(self):
        self.stop_thread = True
        if self.serial_thread:
            self.serial_thread.join()


def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def streaming_loop():
    os.sched_setaffinity(0, {0})
    print(f"[Streaming] running on core: {os.sched_getaffinity(0)}")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def start_processing():
    handler = Mainprocess()
    try:
        handler.processing_loop()
    finally:
        handler.stop()


if __name__ == "__main__":
    process_worker = multiprocessing.Process(target=start_processing)
    process_stream = multiprocessing.Process(target=streaming_loop)

    process_worker.start()
    process_stream.start()

    process_worker.join()
    process_stream.join()
