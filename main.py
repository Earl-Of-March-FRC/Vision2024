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
        x, y, w, h = tuple(map(round, as_np))

        print("TYPE OF X --", type(x))

        cv.rectangle(frame, (x, y), (x+w, h+y), (255, 0, 0), thickness=2)

    cv.imshow("object detection", frame)

    if cv.waitKey(1) == ord("d"):
        break

cap.release()
cv.destroyAllWindows()

"""
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
"""
