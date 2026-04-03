import argparse
import queue
import threading
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
import kornia.geometry as K
import kornia.feature as KF


def expected_angle_step(rpm, fps):
    return (360.0 * rpm) / (60.0 * fps)


def warp_affine(frame, M, size):
    # Convert to tensor
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    M_tensor = torch.from_numpy(M).float().cuda().unsqueeze(0)
    warped_tensor = K.warp_affine(img_tensor, M_tensor, (size[1], size[0]), mode='bilinear', padding_mode='zeros')
    warped = (warped_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(warped)


def estimate_rotation_ecc(prev_gray, curr_gray):
    # Removed: Use predicted rotation only for speed
    return 0.0


def estimate_rotation_ecc_fast(prev_gray, curr_gray, max_dim=480):
    # Removed: Simplified to predicted
    return 0.0


def run_viewer(video_path: str, rpm: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = YOLO("yolo26n.pt")
    draw_detections = True

    print(f"PyTorch CUDA enabled: {torch.cuda.is_available()}")

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
    prev_delta_est = expected_step  # for smoothing

    center = (w // 2, h // 2)

    display_interval = 2  # reduce GUI updates while keeping window alive

    # Telemetry variables
    frame_count = 0
    total_stabilize_time = 0.0
    total_inference_time = 0.0
    start_time = time.time()

    display_queue = queue.Queue(maxsize=10)

    def display_worker():
        while True:
            item = display_queue.get()
            if item is None:
                break
            combined = item
            cv2.imshow("Original (Left) | Stabilized + Vehicles (Right)", combined)
            cv2.waitKey(1)

    display_thread = threading.Thread(target=display_worker, daemon=True)
    display_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- estimate ---
        stabilize_start = time.time()
        delta_est = expected_step  # predicted rotation only for speed

        # --- temporal smoothing (critical) ---
        delta_est = 0.85 * prev_delta_est + 0.15 * delta_est
        prev_delta_est = delta_est

        # --- prediction (ground truth from RPM) ---
        expected_angle += expected_step

        # --- correction (removed for speed) ---
        # correction = delta_est - expected_step
        # ... removed

        # final angle
        cumulative_angle = expected_angle

        # --- stabilize ---
        M = cv2.getRotationMatrix2D(center, -cumulative_angle, 1.0)
        stabilized = warp_affine(frame, M, (w, h))
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
        if frame_count % display_interval == 0:
            display_queue.put(combined)

        prev_gray = gray

    display_queue.put(None)
    display_thread.join()

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