import cv2
import math
import numpy as np

class ObjectDetector:
    def __init__(self, focal_length_x, object_real_width, cascade_model_path):
        self.focal_length_x = focal_length_x
        self.object_real_width = object_real_width
        self.model = cv2.CascadeClassifier(cascade_model_path)
        self.max_idx = 0

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
    
    def text(self, frame: np.ndarray, text: str, idx: int, bbox: tuple) -> None:
        x, y, w, h = bbox
        cv2.putText(    
            frame,
            text=text,
            org=(x, y - (idx * 17)), 
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(0, 0, 255),
            lineType=cv2.LINE_AA
        )


def main():
    focal_length_x = 50
    object_real_width = 5
    cascade_model_path = "path_vision.xml"

    object_detector = ObjectDetector(focal_length_x, object_real_width, cascade_model_path)

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        object_counter = 0
        
        targets = object_detector.detect_objects(frame)
        

        for (x, y, w, h) in targets:
            object_counter += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=5)
            distance = object_detector.calculate_distance_with_offset(w)
            horizontal_angle = object_detector.calculate_horizontal_angle(frame.shape[1], x + w / 2)

            if -15 < horizontal_angle < 15:
                angle_description = "Forward"
            elif horizontal_angle >= 15:
                angle_description = "Right"
            else:
                angle_description = "Left"
            
            print(angle_description)

            object_detector.text(frame,f"Object {object_counter}: Distance: {distance:.2f} inches", 1, (x,y,w,h))
            object_detector.text(frame,f"Horizontal Angle: {horizontal_angle:.2f} degrees", 2, (x,y,w,h))


        cv2.imshow("Vision Processing", frame)
        key = cv2.waitKey(1)
        if key == ord("q") or key == ord("d"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
