import cv2
import math
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results

class ObjectDetector:
    def __init__(self, model_path, focal_length_x, object_real_width, confidence_threshold=0.5) -> None:
        self.focal_length_x = focal_length_x
        self.object_real_width = object_real_width
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def calculate_distance_with_offset(self, object_apparent_width: float) -> float:
        return (lambda distance: distance / 25.4)((self.object_real_width * self.focal_length_x) / object_apparent_width)

    def calculate_horizontal_angle(self, frame_width: float, frame_height: float, object_center_x: float, object_center_y: float) -> float:
        screen_center_x = frame_width / 2
        screen_center_y = frame_height / 2

        # Calculate the displacement of the object's center from the screen center
        delta_x = object_center_x - screen_center_x
        delta_y = object_center_y - screen_center_y  # y-coordinate from top to bottom

        # Calculate the angle displacement using atan2
        angle = math.atan2(delta_y, delta_x)
        
        # Map the angle such that 90 degrees corresponds to 0, and positive/negative angles represent right/left
        mapped_angle = math.degrees(angle) - 90
        if mapped_angle < -180:
            mapped_angle += 360
        elif mapped_angle > 180:
            mapped_angle -= 360

        return mapped_angle


    def detect_objects(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        results = self.model.predict(frame)

        objects = []
        for result in results.xyxy:
            x_center, y_center, w, h = map(round, result[:4])
            objects.append((x_center, y_center, w, h))

        return objects

class ScreenItems:
    def __init__(self) -> None:
        pass

    def text_above(self, frame: np.ndarray, text: str, color: tuple, pos: int, bbox: tuple, thickness: int = 1, scale: int = 1) -> None:
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

    def text_right_up(self, frame: np.ndarray, text: str, color: tuple,) -> None:
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
    focal_length_x = 265.69410448932825 # in mm
    object_real_width =  (lambda distance_in_inches: distance_in_inches * 25.4)(14.875) #in inches, does conversion to mm
    model_path = 'pretrained.pt'

    object_detector = ObjectDetector(model_path, focal_length_x, object_real_width)
    screen_items = ScreenItems()

    cap = cv2.VideoCapture(0)

    
    
    while True:

        
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        object_counter = 0

        results: list[Results] = object_detector.model.predict(frame, stream=True)

        for result in results:
            # object_counter += 1
            as_np: np.ndarray[np.ndarray[np.float32]] = result.boxes.xywh.numpy()
            try:
                bool(as_np)
            except ValueError:
                as_np = as_np[0].tolist()
            else:
                continue

            print("THE ARRAY --", as_np)
            x_center, y_center, w, h = tuple(map(round, as_np))

            # Calculate top-left corner coordinates based on center
            x_top_left = int(x_center - w / 2)
            y_top_left = int(y_center - h / 2)

            cv2.rectangle(frame, (x_top_left, y_top_left), (x_top_left+w, y_top_left+h), (255, 255, 0), thickness=2)

            distance = object_detector.calculate_distance_with_offset(w)
            horizontal_angle = object_detector.calculate_horizontal_angle(frame.shape[1],frame.shape[0],x_center, y_center)

             

            screen_items.text_above(frame,f"Horizontal Angle: {horizontal_angle:.2f} degrees", (255,255,0), 2, (x_top_left,y_top_left,w,h), 2)
            screen_items.text_above(frame,f"Object {object_counter}: Distance: {distance:.2f} inches", (255,255,0), 1, (x_top_left,y_top_left,w,h), 2 )
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
