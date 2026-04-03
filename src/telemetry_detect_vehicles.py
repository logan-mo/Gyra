import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def expected_angle_step(rpm, fps):
    return (360.0 * rpm) / (60.0 * fps)


def is_cuda_available():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def warp_affine(frame, M, size, use_cuda):
    if use_cuda:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            warped = cv2.cuda.warpAffine(
                gpu_frame,
                M,
                size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            return warped.download()
        except Exception:
            pass
    return cv2.warpAffine(
        frame,
        M,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


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
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-4),
        )
    except:
        return 0.0

    angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    return angle


def estimate_rotation_ecc_fast(prev_gray, curr_gray, max_dim=480):
    if max(prev_gray.shape) > max_dim:
        small_prev = cv2.resize(prev_gray, (max_dim, int(prev_gray.shape[0] * max_dim / prev_gray.shape[1])))
        small_curr = cv2.resize(curr_gray, (max_dim, int(curr_gray.shape[0] * max_dim / curr_gray.shape[1])))
    else:
        small_prev = prev_gray
        small_curr = curr_gray

    return estimate_rotation_ecc(small_prev, small_curr)


def run_viewer(video_path: str, rpm: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = YOLO("yolo26n.pt")
    draw_detections = True

    use_cuda = is_cuda_available()
    print(f"OpenCV CUDA enabled: {use_cuda}")

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

    # Telemetry variables
    frame_count = 0
    total_stabilize_time = 0.0
    total_inference_time = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- estimate ---
        stabilize_start = time.time()
        delta_est = estimate_rotation_ecc_fast(prev_gray, gray, max_dim=320)

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
        stabilized = warp_affine(frame, M, (w, h), use_cuda)
        stabilize_end = time.time()
        total_stabilize_time += (stabilize_end - stabilize_start)

        # use a smaller inference input for speed (up to 640x480)
        infer_max_w = 640
        infer_max_h = 480
        scale = min(1.0, infer_max_w / w, infer_max_h / h)
        infer_w = int(w * scale)
        infer_h = int(h * scale)
        infer_frame = cv2.resize(stabilized, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else stabilized

        src_x = w / infer_w
        src_y = h / infer_h

        inference_start = time.time()
        results = model.predict(infer_frame, half=True, verbose=False)[0]
        inference_end = time.time()
        total_inference_time += (inference_end - inference_start)

        if draw_detections:
            for box in results.boxes:
                # get bounding box coordinates on inference frame
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                x1 = int(x1 * src_x)
                y1 = int(y1 * src_y)
                x2 = int(x2 * src_x)
                y2 = int(y2 * src_y)
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

    # Telemetry output
    total_time = time.time() - start_time
    processed_fps = frame_count / total_time if total_time > 0 else 0
    avg_stabilize_time = total_stabilize_time / frame_count if frame_count > 0 else 0
    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0

    print(f"Telemetry:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processed FPS: {processed_fps:.2f}")
    print(f"Average time per frame for stabilization: {avg_stabilize_time:.4f} seconds")
    print(f"Average time per frame for inference: {avg_inference_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path", type=str)
    parser.add_argument("--rpm", required=True, type=int)

    args = parser.parse_args()

    run_viewer(args.video_path, args.rpm)