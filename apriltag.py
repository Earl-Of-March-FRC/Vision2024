import cv2 as cv
import math
import numpy as np
from numpy.typing import NDArray
import robotpy_apriltag as apriltags
from wpimath.geometry import Pose3d, Rotation3d, Transform3d, Translation3d

field = apriltags.loadAprilTagLayoutField(apriltags.AprilTagField.k2024Crescendo)
field.setOrigin(apriltags.AprilTagFieldLayout.OriginPosition.kBlueAllianceWallRightSide)
translation = Translation3d(x=field.getFieldLength() / 2, y=field.getFieldWidth() / 2, z=0)
field.setOrigin(Pose3d().transformBy(Transform3d(translation=translation, rotation=Rotation3d())))
tags = field.getTags()
origin = field.getOrigin()

cap = cv.VideoCapture(0)
detector = apriltags.AprilTagDetector()
detector.addFamily("tag36h11")

estimator_cfg = apriltags.AprilTagPoseEstimator.Config(
    tagSize=0.1651,
    fx=658.8667730937517,
    fy=658.5911798128103,
    cx=324.01396488468737,
    cy=234.71600824324184,
)
pose_estimator = apriltags.AprilTagPoseEstimator(config=estimator_cfg)

MatLike = NDArray[np.uint8]
def text(frame: MatLike, text: str, idx: int) -> None:
    cv.putText(
        frame,
        text=text,
        org=(50, 30 * idx),
        fontFace=cv.FONT_HERSHEY_PLAIN,
        fontScale=2,
        color=(0, 0, 255),
        lineType=cv.LINE_AA
    )

def to_deg(f: float) -> float:
    return f * math.pi / 180

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    targets = detector.detect(gray)
    # print(targets)

    for target in targets:
        estimate = pose_estimator.estimate(target)

        tag_pose = field.getTagPose(target.getId())

        # theoretically shouldn't ever be `None`, but theoretical isn't guaranteed
        if tag_pose is None or target.getId() == 4:
            continue

        tag_rotation = tag_pose.rotation().z_degrees
        perpendicular = tag_rotation + 90
        raa_i_think = abs(180 - perpendicular)
        true_angle = raa_i_think - estimate.rotation().y_degrees
        hypotenuse = math.sqrt((estimate.translation().x ** 2) + (estimate.translation().z ** 2))
        print(true_angle, hypotenuse, to_deg(math.cos(true_angle)), to_deg(math.sin(true_angle)))
        x_offset = to_deg(math.cos(true_angle)) * hypotenuse
        y_offset = to_deg(math.sin(true_angle)) * hypotenuse

        r_x = tag_pose.x + x_offset
        r_the_other_axis = tag_pose.y + y_offset
        # print(estimate)

        to_apriltag = tag_pose.translation().distance(estimate.translation())
        tag_to_origin = tag_pose.translation().distance(origin.translation())
        ro = estimate.rotation()
        # print(f"X: {round(ro.x_degrees, 2)} -- Y: {round(ro.y_degrees, 2)} -- Z: {round(ro.z_degrees, 2)}")
        the_angle = to_deg(tag_pose.relativeTo(origin).rotation().angle)

        missing_side = math.sqrt((to_apriltag ** 2 + tag_to_origin ** 2) - (2 * tag_to_origin * to_apriltag * to_deg(math.cos(the_angle))))

        top_left = target.getCorner(0)
        bottom_right = target.getCorner(2)

        x, y, w, h = (
            round(top_left.x),
            round(top_left.y),
            round(bottom_right.x - top_left.x),
            round(bottom_right.y - top_left.y),
        )

        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        text(frame, f"X: {estimate.x}", idx=1)
        text(frame, f"Y: {estimate.y}", idx=2)
        text(frame, f"Z: {estimate.z}", idx=3)
        text(frame, f"Location: ({r_x:.2f}, {r_the_other_axis:.2f})", idx=4)

    cv.imshow("BALLS", frame)

    if cv.waitKey(1) == ord("d"):
        break

cap.release()
cv.destroyAllWindows()
