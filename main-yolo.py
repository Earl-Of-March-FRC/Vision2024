import cv2 as cv
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Results

# Load a model
model = YOLO('pretrained.pt')  # pretrained YOLOv8n model

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    results: list[Results] = model.predict(frame)

    for result in results:
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

        cv.rectangle(frame, (x_top_left, y_top_left), (x_top_left+w, y_top_left+h), (255, 255, 0), thickness=2)

    cv.imshow("object detection", frame)

    if cv.waitKey(1) == ord("d"):
        break

cap.release()
cv.destroyAllWindows()
