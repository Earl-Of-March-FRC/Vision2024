import cv2
import math
import numpy as np
class ObjectDetector:
    def __init__(self, focal_length_x, object_real_width, cascade_model_path):
        self.focal_length_x = focal_length_x
        self.object_real_width = object_real_width
        self.model = cv2.CascadeClassifier(cascade_model_path)

    def calculate_distance_with_offset(self, object_apparent_width: float) -> float:
        return self.convert_units_to_inches((self.object_real_width * self.focal_length_x) / object_apparent_width)

    def calculate_horizontal_angle(self, frame_width: float, object_center_x: float) -> float:
        screen_center_x = frame_width / 2
        angle = math.atan((object_center_x - screen_center_x) / self.focal_length_x)
        return math.degrees(angle)

    def detect_objects(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.model.detectMultiScale(gray, 1.3, 5)
    
    def convert_units_to_inches(self, distance_in_mm: float) -> float:
        return distance_in_mm / 25.4


def main():
    focal_length_x = 50
    object_real_width = 5
    cascade_model_path = "path-to-model.xml"

    object_detector = ObjectDetector(focal_length_x, object_real_width, cascade_model_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        targets = object_detector.detect_objects(frame)
        
        for (x, y, w, h) in targets:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=5)
            distance = object_detector.calculate_distance_with_offset(w)
            horizontal_angle = object_detector.calculate_horizontal_angle(frame.shape[1], x + w / 2)
            cv2.putText(frame, f"Distance to object: {distance:.2f} inches", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Horizontal Angle: {horizontal_angle:.2f} degrees", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Cascade Detector", frame)
        key = cv2.waitKey(1)
        if key == ord("q") or key == ord("d"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
