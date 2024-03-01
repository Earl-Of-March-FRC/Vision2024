from __future__ import annotations

import cv2
import math
import ntcore
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results

from typing import TypeVar, TYPE_CHECKING

from multiprocessing import Queue

T = TypeVar("T")
MatLike = np.ndarray[T]

# THIS IS FOR MY OWN WEBCAM, REMEMBER TO CALIBRATE THE CAMERA AND REPLACE THIS
cam_matrix = np.array([[658.86677309, 0, 324.01396488], [0, 658.59117981, 234.71600824], [0, 0, 1]])

class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        focal_length_x: float,
        object_real_width: float,
        confidence_threshold: float = 0.5
    ) -> None:
        self.focal_length_x = focal_length_x
        self.object_real_width = object_real_width
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def calculate_distance_with_offset(self, detection_width: float) -> float:
        return (lambda distance: distance / 25.4)((self.object_real_width * self.focal_length_x) / detection_width)

    def calculate_horizontal_angle(self, frame: MatLike, object_center_x: float) -> float:
        """
        https://stackoverflow.com/questions/55080775/opencv-calculate-angle-between-camera-and-object my beloved
        """
        screen_center_x = frame.shape[1] / 2
        screen_center_y = frame.shape[0] / 2

        mat_inverted = np.linalg.inv(cam_matrix)
        vector1: MatLike = mat_inverted.dot((object_center_x, screen_center_y, 1.0))
        vector2: MatLike = mat_inverted.dot((screen_center_x, screen_center_y, 1.0))
        cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cos_angle = max(-1, min(cos_angle, 1)) #Esures cos_angle is within the valid range (-1,1); no crashes
        real_angle = math.degrees(math.acos(cos_angle))

        if object_center_x < screen_center_x:
            real_angle *= -1

        return real_angle
    
class Worker(ObjectDetector):
    def __init__(
            self, 
            model_path: str, 
            focal_length_x: float, 
            object_real_width: float
        ) -> None:
        super().__init__(model_path, focal_length_x, object_real_width)

    def process_frame(self, frame: MatLike) -> MatLike:
        results: list[Results] = self.model.predict(frame)
        processed_frame = frame.copy()
        controller = NetworkTablesController()  
        for result in results:
            as_np: np.ndarray[np.ndarray[np.float32]] = result.boxes.xywh.numpy()
            try:
                bool(as_np)
            except ValueError:
                as_np = as_np[0].tolist()
            else:
                continue

            x_center, y_center, w, h = tuple(map(round, as_np))

            x_left = int(x_center - w / 2)
            y_top = int(y_center - h / 2)

            cv2.rectangle(processed_frame, (x_left, y_top), (x_left+w, y_top+h), (255, 255, 0), thickness=2)

            angle = self.calculate_horizontal_angle(frame, x_center)
            distance_object = self.calculate_distance_with_offset(w)

            controller.send_data(angle,distance_object)

            ScreenItems.text_above(processed_frame,f"Horizontal Angle: {angle:.2f} degrees", (255,255,0), 2, (x_left,y_top,w,h), 2)
            ScreenItems.text_above(processed_frame,f"Distance: {distance_object:.2f} in", (255,255,0), 1, (x_left,y_top,w,h), 2)

            processed_frame = cv2.circle(processed_frame, (x_center, y_center), 1, (0,255,0), 2)

        return processed_frame




class ScreenItems:
    @staticmethod
    def text_above(frame: MatLike, text: str, color: tuple, pos: int, bbox: tuple, thickness: int = 1, scale: int = 1) -> None:
        x, y, w, h = bbox
        cv2.putText(
            frame,
            text=text,
            org=(x, y - (pos * 17)),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale= scale,
            color= color,
            lineType=cv2.LINE_AA,
            thickness=thickness
        )

    @staticmethod
    def text_right_up(frame: MatLike, text: str, color: tuple) -> None:
        height, width, _ = frame.shape

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_width, text_height = text_size

        cv2.putText(
            frame,
            text=text,
            org=(width - text_width - 120, 30),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale= 2,
            thickness= 1,
            color= color,
            lineType=cv2.LINE_AA
        )

class NetworkTablesController:
    def __init__(self) -> None:
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.table = self.inst.getTable("datatable")
        self.distance_pub = self.table.getDoubleTopic("distance").publish()
        self.angle_pub = self.table.getDoubleTopic("angle").publish()
        self.inst.startClient4("example client")
        self.inst.setServer("localhost")
        self.inst.setServerTeam(7476, 0)
        self.inst.startDSClient()

    def send_data(self, angle: float, distance: float) -> None:
        self.angle_pub.set(angle)
        self.distance_pub.set(distance)
        print(f"Angle: {angle}, Distance: {distance}")

def main():

    focal_length_x = 658.867 # in mm
    object_real_width =  (lambda distance_in_inches: distance_in_inches * 25.4)(14.875) #in inches, does conversion to mm
    model_path = 'pretrained.pt'

    input_q = Queue()
    output_q = Queue()

    worker = Worker(model_path, focal_length_x, object_real_width)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_q.put(frame)
        processed_frame = worker.process_frame(input_q.get())
        output_q.put(processed_frame)
        cv2.imshow("object detection", processed_frame)
        if cv2.waitKey(1) == ord("d"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
