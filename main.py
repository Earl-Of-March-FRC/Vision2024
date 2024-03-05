from __future__ import annotations

import cv2
import logging
import math
# import time
import numpy as np

import ntcore
from ultralytics import YOLO
from ultralytics.engine.results import Results

from mjpeg_streamer import MjpegServer, Stream

from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    T = TypeVar("T")
    MatLike = np.ndarray[T]

logging.basicConfig(level=logging.DEBUG)

# THIS IS FOR MY OWN WEBCAM, REMEMBER TO CALIBRATE THE CAMERA AND REPLACE THIS
cam_matrix = np.array([[508.13950658192897, 0, 315.83545490013387], 
                       [0, 508.4437534984872, 244.77465580560457], 
                       [0, 0, 1]])

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

    def calculate_horizontal_angle(self, frame: MatLike, /, object_center_x: float, camera_offset: float) -> float:
        """
        Calculate the horizontal angle between the camera and the object.
        """
        screen_center_x = frame.shape[1] / 2
        screen_center_y = frame.shape[0] / 2

        # Adjust the object center x-coordinate based on camera offset
        object_center_x -= camera_offset

        mat_inverted = np.linalg.inv(cam_matrix)
        vector1: MatLike = mat_inverted.dot((object_center_x, screen_center_y, 1.0))
        vector2: MatLike = mat_inverted.dot((screen_center_x, screen_center_y, 1.0))
        cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        real_angle = math.degrees(math.acos(cos_angle))

        if object_center_x < screen_center_x:
            real_angle *= -1

        return real_angle


    def cropped(self, frame: MatLike, /, top_left_x: int, top_left_y: int, new_width: int, new_height: int) -> MatLike:
        return frame[top_left_y:top_left_y+new_height,top_left_x:top_left_x+new_width]

    def detect_objects(self, frame: MatLike, /) -> list[tuple[int,int,int,int]]:
        results = self.model.predict(frame)

        objects = []
        for result in results.xyxy:
            x_center, y_center, w, h = map(round, result[:4])
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

class NetworkTable:
    def __init__(self, *, instance: ntcore.NetworkTableInstance | None = None):
        """
        READ THIS: https://docs.wpilib.org/en/stable/docs/software/networktables/networktables-networking.html

        The robot is the server, therefore this script is the client, and so this table instance should be set up as such
        """

        self._inst = instance or ntcore.NetworkTableInstance.getDefault()
        self._table = self._inst.getTable("vision")
        self._distance = self._table.getDoubleTopic("distance").publish()
        self._angle = self._table.getDoubleTopic("angle").publish()

        self._inst.startClient4("vision client")
        # self._inst.setServerTeam(7476, 0)
        self._inst.setServer("10.74.76.227", port=ntcore.NetworkTableInstance.kDefaultPort4)

    def send_data(self, distance: float, angle: float) -> None:
        self._distance.set(distance)
        self._angle.set(angle)
        logging.debug("Angle: %.2f, Distance: %.2f", angle, distance)

    def close(self):
        self._inst.stopClient()

    @property
    def instance(self) -> ntcore.NetworkTableInstance:
        return self._inst

cap = cv2.VideoCapture(0)
stream = Stream("driverfeed", size=(640, 480), quality=50, fps=30)
server = MjpegServer("10.74.76.69", 8080)
ntable = NetworkTable()

def main():
    focal_length_x = cam_matrix[0][0] # in mm
    object_real_width =  (lambda distance_in_inches: distance_in_inches * 25.4)(14.875) #in inches, does conversion to mm

    model_path = "pretrained.pt"
    object_detector = ObjectDetector(model_path, focal_length_x, object_real_width)

    server.add_stream(stream)
    server.start()

    # fps = 0
    # frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.critical("COULD NOT READ FRAME")

        stream.set_frame(frame)
        frame = frame[288:480, 0:frame.shape[1]]
        # frame = object_detector.cropped(frame, top_left_x=0, top_left_y=288, new_width=frame.shape[1], new_height=192)

        object_counter = 0

        # start = time.monotonic()
        results: list[Results] = object_detector.model.predict(frame, imgsz=(640, 192), vid_stride=10, max_det=1, conf=0.2)
        # results = []
        # print("DIMENSIONS IS ", frame.shape)
        # end = time.monotonic()

        # frame_count += 1
        # fps += 1/(end-start or 0.00001)
        # print("FRAMES PER SECOND --", fps / frame_count)

        for result in results:
            # """
            as_np: np.ndarray[np.ndarray[np.float32]] = result.boxes.xywh.numpy()
            try:
                bool(as_np)
            except ValueError:
                as_np = as_np[0].tolist()
            else:
                ntable.send_data(-1, -1)
                continue

            object_counter += 1
            x_center, y_center, w, h = tuple(map(round, as_np))

            # Calculate top-left corner coordinates based on center
            x_left = int(x_center - w / 2)
            y_top = int(y_center - h / 2)

            cv2.rectangle(frame, (x_left, y_top), (x_left+w, y_top+h), (255, 255, 0), thickness=2)

            distance = object_detector.calculate_distance_with_offset(w)
            angle = object_detector.calculate_horizontal_angle(frame, x_center, 0)

            ntable.send_data(distance, angle)

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
            # """

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
