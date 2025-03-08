import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class LivenessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Liveness Detection - Team Beginners")

        self.camera_index = 0
        self.variance_threshold = 50
        self.running = False

        self.cap = None

        # Main Frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Headline Frame
        headline_frame = ttk.Frame(main_frame)
        headline_frame.pack(pady=10)

        headline_label = ttk.Label(headline_frame, text="Face Liveness Detection", font=("Helvetica", 24, "bold"))
        headline_label.pack()

        welcome_label = ttk.Label(headline_frame, text="Welcome to Team Beginners' Liveness Detection System", font=("Helvetica", 12))
        welcome_label.pack()

        # Video Frame
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(pady=10)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        # Control Frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)

        self.start_button = ttk.Button(control_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Threshold Frame
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.pack(pady=10)

        self.threshold_label = ttk.Label(threshold_frame, text="Threshold:")
        self.threshold_label.pack(side=tk.LEFT, padx=5)

        self.threshold_scale = ttk.Scale(threshold_frame, from_=10, to=150, orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_scale.set(self.variance_threshold)
        self.threshold_scale.pack(side=tk.LEFT, padx=5)

        # Result Frame
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(pady=10)

        self.result_label = ttk.Label(result_frame, text="Detection Result: Waiting...", font=("Helvetica", 12))
        self.result_label.pack()

    def update_threshold(self, value):
        self.variance_threshold = int(float(value))

    def start_detection(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.cap = cv2.VideoCapture(self.camera_index)
            threading.Thread(target=self.detect_liveness).start()

    def stop_detection(self):
        if self.running:
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if self.cap and self.cap.isOpened():
                self.cap.release()

    def detect_liveness(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            result_text = "Waiting..."

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                variance = cv2.Laplacian(face_roi, cv2.CV_64F).var()

                text = "Live" if variance > self.variance_threshold else "Spoof"
                color = (0, 255, 0) if variance > self.variance_threshold else (0, 0, 255)

                cv2.putText(frame, f"{text} (Variance: {variance:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                result_text = f"Detection Result: {text} (Variance: {variance:.2f})"

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)
            self.result_label.config(text=result_text)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.video_label.config(image='')
            self.result_label.config(text="Detection Result: Stopped.")

def main():
    root = tk.Tk()
    app = LivenessDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()