import argparse
import collections
import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from vidstab import VidStab


def center_crop(frame: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    h, w = frame.shape[:2]
    if h == target_h and w == target_w:
        return frame

    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    cropped = frame[top:top + target_h, left:left + target_w]

    if cropped.shape[:2] != (target_h, target_w):
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return cropped


def run_viewer(
    video_path: str,
    rpm: int,
    output_path: str = None,
    smoothing_window: int = 30,
    border_type: str = "black",
    border_size: int = 150,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = YOLO("yolo26n.pt")
    stabilizer = VidStab(processing_max_dim=640)

    print(f"PyTorch CUDA enabled: {torch.cuda.is_available()}")
    print("Using vidstab runtime stabilization")
    print("Running ultralytics model.predict path")
    print(f"Input RPM set to: {rpm}")
    print(f"Smoothing window: {smoothing_window}, border type: {border_type}, border size: {border_size}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine input video FPS.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    display_interval = 2
    frame_count = 0
    total_inference_time = 0.0
    start_time = time.time()

    original_queue = collections.deque()

    video_writer = None
    if output_path:
        combined_w = 2 * w
        combined_h = h
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))
        if not video_writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open output video: {output_path}")
        print(f"Saving output video to: {output_path}")

    display_queue = queue.Queue(maxsize=10)

    def display_worker():
        while True:
            item = display_queue.get()
            if item is None:
                break
            cv2.imshow("Original (Left) | Stabilized + Vehicles (Right)", item)
            cv2.waitKey(1)

    display_thread = None
    if not output_path:
        display_thread = threading.Thread(target=display_worker, daemon=True)
        display_thread.start()

    def process_frame(original_frame: np.ndarray, stabilized_frame: np.ndarray):
        nonlocal frame_count, total_inference_time

        stabilized_frame = center_crop(stabilized_frame, (h, w))

        if np.any(stabilized_frame):
            infer_max_w = 640
            infer_max_h = 480
            scale = min(1.0, infer_max_w / w, infer_max_h / h)
            infer_w = int(w * scale)
            infer_h = int(h * scale)
            infer_frame = (
                cv2.resize(stabilized_frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                if scale < 1.0
                else stabilized_frame
            )

            inference_start = time.time()
            results = model.predict(infer_frame, half=True, verbose=False)[0]
            inference_end = time.time()
            total_inference_time += inference_end - inference_start

            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                x1 = int(x1 * (w / infer_w))
                y1 = int(y1 * (h / infer_h))
                x2 = int(x2 * (w / infer_w))
                y2 = int(y2 * (h / infer_h))
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls in [2, 5, 7]:
                    cv2.rectangle(stabilized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        stabilized_frame,
                        f"{results.names[cls]} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        rpm_text = f"RPM: {rpm}"
        cv2.putText(
            stabilized_frame,
            rpm_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined = np.hstack((original_frame, stabilized_frame))
        if output_path:
            video_writer.write(combined)
        else:
            if frame_count % display_interval == 0:
                display_queue.put(combined)

    while True:
        grabbed, original_frame = cap.read()
        if not grabbed:
            break

        original_queue.append(original_frame)
        stabilized_frame = stabilizer.stabilize_frame(
            input_frame=original_frame,
            smoothing_window=smoothing_window,
            border_type=border_type,
            border_size=border_size,
        )

        if stabilized_frame is None:
            break

        if len(original_queue) <= smoothing_window:
            continue

        delayed_original = original_queue.popleft()
        frame_count += 1
        process_frame(delayed_original, stabilized_frame)

    while True:
        stabilized_frame = stabilizer.stabilize_frame(
            input_frame=None,
            smoothing_window=smoothing_window,
            border_type=border_type,
            border_size=border_size,
        )
        if stabilized_frame is None:
            break

        if not original_queue:
            break

        delayed_original = original_queue.popleft()
        frame_count += 1
        process_frame(delayed_original, stabilized_frame)

    if output_path and video_writer is not None:
        video_writer.release()
    elif display_thread is not None:
        display_queue.put(None)
        display_thread.join()

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    processed_fps = frame_count / total_time if total_time > 0 else 0
    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0

    print("Telemetry:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processed FPS: {processed_fps:.2f}")
    print(f"Average time per frame for inference: {avg_inference_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video_path", type=str)
    parser.add_argument("--rpm", required=True, type=int)
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output video (e.g., output.mp4). If not provided, displays the video.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=30,
        help="Number of frames used for smoothing stabilization.",
    )
    parser.add_argument(
        "--border-size",
        type=int,
        default=500,
        help="Amount of border padding added before stabilization.",
    )
    parser.add_argument(
        "--border-type",
        type=str,
        default="black",
        choices=["black", "reflect", "replicate"],
        help="Border fill mode used by stabilization.",
    )

    args = parser.parse_args()

    if args.output:
        output_folder = os.path.dirname(args.output) or "."
        if not os.path.exists(output_folder):
            raise FileNotFoundError("Destination folder doesn't exist")

    run_viewer(
        args.video_path,
        args.rpm,
        args.output,
        smoothing_window=args.smoothing_window,
        border_type=args.border_type,
        border_size=args.border_size,
    )
