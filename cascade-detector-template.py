# starter code for a cascade detector

import cv2

model = cv2.CascadeClassifier("path-to-model.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    targets = model.detectMultiScale(gray, 1.3, 5) # change these params as needed

    for (x, y, w, h) in targets:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=5)

    cv2.imshow("balls", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
