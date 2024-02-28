import cv2
import math
import numpy as np

from networktables import NetworkTablesController


class ObjectDetector:
    def __init__(self, focal_length_x, object_real_width, cascade_model_path) -> None:
        self.focal_length_x = focal_length_x
        self.object_real_width = object_real_width
        self.model = cv2.CascadeClassifier(cascade_model_path)
        self.max_idx = 0

    def calculate_distance_with_offset(self, object_apparent_width: float) -> float:
        return (lambda distance: distance / 25.4)((self.object_real_width * self.focal_length_x) / object_apparent_width)

    def calculate_horizontal_angle(self, frame_width: float, object_center_x: float) -> float:
        screen_center_x = frame_width / 2
        angle = math.atan((object_center_x - screen_center_x) / self.focal_length_x)
        return math.degrees(angle) # Angle between horizontal and camera

    def detect_objects(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.model.detectMultiScale(gray, 1.3, 5)

    
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

    def text_left_bottom(self, frame: np.ndarray, text: str, color: tuple,) -> None:
        height, width, _ = frame.shape
        cv2.putText(    
            frame,
            text=text,
            org=(10, height - 20), 
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale= 2,
            thickness= 1,
            color= color, 
            lineType=cv2.LINE_AA
        )
    def text_right_bottom(self, frame: np.ndarray, text: str, color: tuple,) -> None:
        height, width, _ = frame.shape
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_width, text_height = text_size

        cv2.putText(    
            frame,
            text=text,
            org=(width - text_width - 120, height - 20), 
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale= 2,
            thickness= 1,
            color= color, 
            lineType=cv2.LINE_AA
        )
    def text_left_up(self, frame: np.ndarray, text: str, color: tuple,) -> None:
        height, width, _ = frame.shape
        cv2.putText(    
            frame,
            text=text,
            org=(5, 30), 
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale= 2,
            thickness= 1,
            color= color, 
            lineType=cv2.LINE_AA
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
    focal_length_x = 50 # in mm
    object_real_width =  (lambda distance_in_inches: distance_in_inches * 25.4)(14.875) #in inches, does conversion to mm
    cascade_model_path = "path_vision.xml"

    object_detector = ObjectDetector(focal_length_x, object_real_width, cascade_model_path)
    screen_items = ScreenItems()
    network_tables_controller = NetworkTablesController()  


    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        object_counter = 0
        
        targets = object_detector.detect_objects(frame)
        

        for (x, y, w, h) in targets:

            object_counter += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=5)
            distance = object_detector.calculate_distance_with_offset(w)
            horizontal_angle = object_detector.calculate_horizontal_angle(frame.shape[1], x + w / 2)

            if -15 < horizontal_angle < 15:
                angle_description = "Forward"
            elif horizontal_angle >= 15:
                angle_description = "Right"
            else:
                angle_description = "Left"
            
            print(angle_description) # Just for us to see which way the robot needs to turn (for us its the camera moving on its horizontal pivot axis)

            screen_items.text_above(frame,f"Horizontal Angle: {horizontal_angle:.2f} degrees", (255,255,0), 2, (x,y,w,h), 2)
            screen_items.text_above(frame,f"Object {object_counter}: Distance: {distance:.2f} inches", (255,255,0), 1, (x,y,w,h), 2 )
            screen_items.text_right_up(frame,f"Move {angle_description}", (255,255,0))

            network_tables_controller.send_data(horizontal_angle, distance)



        cv2.imshow("Vision Processing", frame)
        key = cv2.waitKey(1)
        if key == ord("q") or key == ord("d"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
