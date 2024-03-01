from __future__ import annotations

import cv2
import math
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
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
        real_angle = math.degrees(math.acos(cos_angle))

        if object_center_x < screen_center_x:
            real_angle *= -1

        return real_angle


    def detect_objects(self, frame: MatLike) -> list[tuple[int,int,int,int]]:
        results = self.model.predict(frame)

        objects = []
        for result in results.xyxy:
            x_center, y_center, w, h = map(round, result[:4])
            objects.append((x_center, y_center, w, h))

        return objects

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

def main():
    focal_length_x = 658.867 # in mm
    object_real_width =  (lambda distance_in_inches: distance_in_inches * 25.4)(14.875) #in inches, does conversion to mm
    model_path = 'pretrained.pt'

    object_detector = ObjectDetector(model_path, focal_length_x, object_real_width)

    cap = cv2.VideoCapture(0)



    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        object_counter = 0

        results: list[Results] = object_detector.model.predict(frame)

        for result in results:
            # object_counter += 1
            as_np: np.ndarray[np.ndarray[np.float32]] = result.boxes.xywh.numpy()
            try:
                bool(as_np)
            except ValueError:
                as_np = as_np[0].tolist()
            else:
                continue

            x_center, y_center, w, h = tuple(map(round, as_np))

            # Calculate top-left corner coordinates based on center
            x_left = int(x_center - w / 2)
            y_top = int(y_center - h / 2)

            cv2.rectangle(frame, (x_left, y_top), (x_left+w, y_top+h), (255, 255, 0), thickness=2)

            distance = object_detector.calculate_distance_with_offset(w)
            angle = object_detector.calculate_horizontal_angle(frame, x_center)

            ScreenItems.text_above(frame,f"Horizontal Angle: {angle:.2f} degrees", (255,255,0), 2, (x_left,y_top,w,h), 2)
            ScreenItems.text_above(frame,f"Object {object_counter}: Distance: {distance:.2f} in", (255,255,0), 1, (x_left,y_top,w,h), 2 )
            # screen_items.text_right_up(frame,f"Move {angle_description}", (255,255,0))
            radius = 1
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2
            frame = cv2.circle(frame, (x_center, y_center), radius, color, thickness)

            line_color = (0, 0, 255)  # Red color in BGR
            line_thickness = 2
            cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), line_color, line_thickness)

        cv2.imshow("object detection", frame)

        if cv2.waitKey(1) == ord("d"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
