import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def expected_angle_step(rpm, fps):
    return (360.0 * rpm) / (60.0 * fps)


def estimate_rotation(prev_gray, curr_gray):
    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        return 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    M, _ = cv2.estimateAffinePartial2D(pts1, pts2)

    if M is None:
        return 0.0

    # Extract rotation angle
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    return angle


def estimate_rotation_ecc(prev_gray, curr_gray):
    warp = np.eye(2, 3, dtype=np.float32)

    try:
        _, warp = cv2.findTransformECC(
            prev_gray,
            curr_gray,
            warp,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6),
        )
    except:
        return 0.0

    angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    return angle


def run_viewer(video_path: str, rpm: int):
    cap = cv2.VideoCapture(video_path)

    model = YOLO("yolov8n.pt")  # choose appropriate YOLOv8 model
    draw_detections = True

    if not cap.isOpened():
        raise ValueError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, prev = cap.read()
    if not ret:
        return

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    expected_step = (360.0 * rpm) / (60.0 * fps)

    expected_angle = 0.0
    error = 0.0
    prev_delta_est = expected_step  # for smoothing

    alpha = 0.08  # trust vision VERY lightly
    max_correction = 1.5  # deg per frame (tight clamp)
    max_error = 3.0  # total accumulated correction cap
    center = (w // 2, h // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- estimate ---
        delta_est = estimate_rotation_ecc(prev_gray, gray)

        # --- reject garbage (ECC can spike under blur) ---
        if abs(delta_est) > 30:  # unrealistic jump → ignore
            delta_est = prev_delta_est

        # --- temporal smoothing (critical) ---
        delta_est = 0.85 * prev_delta_est + 0.15 * delta_est
        prev_delta_est = delta_est

        # --- prediction (ground truth from RPM) ---
        expected_angle += expected_step

        # --- correction (phase error only) ---
        correction = delta_est - expected_step

        # clamp per-frame correction
        correction = np.clip(correction, -max_correction, max_correction)

        # integrate SMALL correction only
        error += alpha * correction

        # anti-windup (prevents long-term drift)
        error = np.clip(error, -max_error, max_error)

        # final angle
        cumulative_angle = expected_angle + error

        # --- stabilize ---
        M = cv2.getRotationMatrix2D(center, -cumulative_angle, 1.0)
        stabilized = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        results = model.predict(stabilized, verbose=False)[0]

        if draw_detections:
            for box in results.boxes:
                # get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # only draw vehicles (YOLO COCO: car=2, bus=5, truck=7)
                if cls in [2, 5, 7]:
                    cv2.rectangle(stabilized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        stabilized,
                        f"{results.names[cls]} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        combined = np.hstack((frame, stabilized))
        cv2.imshow("Original (Left) | Stabilized + Vehicles (Right)", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path", type=str)
    parser.add_argument("--rpm", required=True, type=int)

    args = parser.parse_args()

    run_viewer(args.video_path, args.rpm)
