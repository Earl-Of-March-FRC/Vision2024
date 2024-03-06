from __future__ import annotations

import cv2
import logging
import math
import numpy as np

from typing import TypeVar, TYPE_CHECKING

from ultralytics import YOLO
from ultralytics.engine.results import Results

from networkTable import NetworkTable

from mjpeg_streamer import MjpegServer, Stream

if TYPE_CHECKING:
    T = TypeVar("T")
    MatLike = np.ndarray[T]

#Elliott's config
CAM_MATRIX = np.array([[508.13950658192897, 0, 315.83545490013387], 
                       [0, 508.4437534984872, 244.77465580560457], 
                       [0, 0, 1]])

"""
Haoyan's camera config: CAM_MATRIX = np.array([[658.86677309, 0, 324.01396488], 
[0, 658.59117981, 234.71600824], 
[0, 0, 1]])

"""

class ObjectDetector:
    def __init__(
        self,
        model_path: str,
        focal_length_x: float,
        object_real_width: float,
        confidence_threshold: float = 0.5
    ) -> None:
        """
        Initializes the ObjectDetector with necessary parameters.
        """
        self.focal_length_x = focal_length_x
        self.object_real_width = object_real_width
        self.model = YOLO(model_path)  # Initialize YOLO model
        self.confidence_threshold = confidence_threshold

    def calculate_distance_with_offset(self, detection_width: float) -> float:
        """
        Calculates the distance to the object with a given detection width.
        """
        return (lambda distance: distance / 25.4)((self.object_real_width * self.focal_length_x) / detection_width)

    def calculate_horizontal_angle(self, frame: MatLike, /, object_center_x: float, camera_offset: float) -> float:
        """
        Calculate the horizontal angle between the camera and the object.
        """
        try:
            screen_center_x = frame.shape[1] / 2
            screen_center_y = frame.shape[0] / 2

            # Adjust the object center x-coordinate based on camera offset
            object_center_x -= camera_offset

            mat_inverted = np.linalg.inv(CAM_MATRIX)  # Invert camera matrix
            vector1: MatLike = mat_inverted.dot((object_center_x, screen_center_y, 1.0))  # Calculate vector 1
            vector2: MatLike = mat_inverted.dot((screen_center_x, screen_center_y, 1.0))  # Calculate vector 2

            # Handle division by zero when both vectors are zero vectors (will not crash)
            if np.all(vector1 == 0) and np.all(vector2 == 0):
                return 0.0

            cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            real_angle = math.degrees(math.acos(cos_angle))

            if object_center_x < screen_center_x:
                real_angle *= -1

            return real_angle

        except Exception as e:
            logging.error("Error occurred while calculating horizontal angle: %s", e)
            return 0.0

    def cropped(self, frame: MatLike, /, top_left_x: int, top_left_y: int, new_width: int, new_height: int) -> MatLike:
        """
        Returns a cropped portion of the frame.
        """
        return frame[top_left_y:top_left_y+new_height,top_left_x:top_left_x+new_width]

    def detect_objects(self, frame: MatLike, /) -> list[tuple[int,int,int,int]]:
        """
        Detects objects in the given frame.
        """
        results = self.model.predict(frame)  # Get predictions from YOLO model

        objects = []
        for result in results.xyxy:
            x_center, y_center, w, h = map(round, result[:4])  # Extract object coordinates
            objects.append((x_center, y_center, w, h))

        return objects


class ScreenItems:
    @staticmethod
    def text_above(frame: MatLike, /, text: str, color: tuple, pos: int, bbox: tuple, thickness: int = 1, scale: int = 1) -> None:
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
    def text_right_up(frame: MatLike, /, text: str, color: tuple) -> None:
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


# Inital setup of video, server and network tables
cap = cv2.VideoCapture(0)
stream = Stream("driverfeed", size=(640, 480), quality=50, fps=30)
server = MjpegServer("10.74.76.69", 8080)
ntable = NetworkTable()

def main():
    # Calculation of focal length and object real width
    focal_length_x = CAM_MATRIX[0][0]  # in mm
    object_real_width = (lambda distance_in_inches: distance_in_inches * 25.4)(14.875)  # in inches, conversion to mm

    # Initialization of ObjectDetector
    model_path = "pretrained.pt"
    object_detector = ObjectDetector(model_path, focal_length_x, object_real_width)

    # Add stream to server and start it
    server.add_stream(stream)
    server.start()

    while True:
        # Read frame from the camera
        ret, frame = cap.read()
        if not ret:
            logging.critical("COULD NOT READ FRAME")

        # Set frame to the stream and crop it
        stream.set_frame(frame)
        frame = frame[120:480, 0:frame.shape[1]]

        # Predict objects in the frame
        results: list[Results] = object_detector.model.predict(frame, imgsz=(640, 192), vid_stride=10, max_det=1, conf=0.2)

        for result in results:
            as_np: np.ndarray[np.ndarray[np.float32]] = result.boxes.xywh.numpy()
            try:
                bool(as_np)
            except ValueError:
                as_np = as_np[0].tolist()
            else:
                ntable.send_data(-1, -1)
                continue

            x_center, y_center, w, h = tuple(map(round, as_np))

            x_left = int(x_center - w / 2)
            y_top = int(y_center - h / 2)

            # Draw rectangle around the detected object
            cv2.rectangle(frame, (x_left, y_top), (x_left+w, y_top+h), (255, 255, 0), thickness=2)

            # Calculate distance and angle
            distance = object_detector.calculate_distance_with_offset(w)
            angle = object_detector.calculate_horizontal_angle(frame, x_center, 0)

            # Send distance and angle data to NetworkTable
            ntable.send_data(distance, angle)

            # Display distance and angle on the frame
            ScreenItems.text_above(frame,f"Horizontal Angle: {angle:.2f} degrees", (255,255,0), 2, (x_left,y_top,w,h), 2)
            ScreenItems.text_above(frame,f"Distance: {distance:.2f} in", (255,255,0), 1, (x_left,y_top,w,h), 2 )
            
            # Draw circle and line when object is on screen for easy angle and distance viewing
            cv2.circle(frame, (x_center, y_center), 1, (0, 255, 0), 2)
            cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (0, 0, 255), 2)

        cv2.imshow("object detection", frame)

        if cv2.waitKey(1) == ord("d"):
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        server.stop()
        ntable.close()
        cap.release()
        cv2.destroyAllWindows()
