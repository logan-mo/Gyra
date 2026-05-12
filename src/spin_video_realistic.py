import os
import math
import argparse
from pathlib import Path

import cv2
import numpy as np


def get_per_frame_rotation(rpm: float, fps: float) -> float:
    """Return degrees to rotate per frame for a given RPM."""
    return (360.0 * rpm) / (60.0 * fps)


def warp(frame: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def apply_motion_blur(frame: np.ndarray, base_angle: float, angle_step: float, samples: int = 5) -> np.ndarray:
    """
    Simulates rotational motion blur by integrating multiple sub-rotations.
    """
    h, w = frame.shape[:2]
    acc = np.zeros_like(frame, dtype=np.float32)

    for i in range(samples):
        a = base_angle + (i / samples) * angle_step
        M = cv2.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
        acc += warp(frame, M, w, h)

    return (acc / samples).astype(np.uint8)


def draw_rpm_overlay(frame: np.ndarray, rpm_value: float) -> None:
    """Draw RPM text in the top-right corner of the frame."""
    text = f"RPM: {rpm_value:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    margin = 10

    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size
    y = margin + text_height
    x = frame.shape[1] - text_width - margin

    cv2.rectangle(
        frame,
        (x - 6, y - text_height - 6),
        (x + text_width + 6, y + baseline + 6),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def eased_fraction(x: float) -> float:
    """Smooth ease-in for the acceleration profile."""
    return x * x * (3.0 - 2.0 * x)


def get_rpm_profile(
    elapsed_seconds: float,
    target_rpm: float,
    accel_duration: float,
    oscillation_factor: float,
    oscillation_period: float,
) -> float:
    """Compute the current RPM from the acceleration and oscillation profile."""
    if elapsed_seconds <= accel_duration:
        return target_rpm * eased_fraction(min(elapsed_seconds / accel_duration, 1.0))

    steady_time = elapsed_seconds - accel_duration
    base_rpm = target_rpm
    amplitude = target_rpm * oscillation_factor
    slow_modulation = 0.6 + 0.4 * math.sin(2.0 * math.pi * steady_time / max(oscillation_period * 3.0, 1e-3))
    oscillation = math.sin(2.0 * math.pi * steady_time / max(oscillation_period, 1e-3)) * amplitude * slow_modulation
    return base_rpm + oscillation


def rotate_video(
    src_path: str,
    destination_path: str,
    rpm: int,
    blur_samples: int,
    accel_duration: float,
    oscillation_factor: float,
    oscillation_period: float,
) -> None:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not determine input video FPS.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(destination_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video: {destination_path}")

    current_angle = 0.0
    frame_index = 0
    center = (w // 2, h // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_seconds = frame_index / fps
        current_rpm = get_rpm_profile(
            elapsed_seconds,
            float(rpm),
            accel_duration,
            oscillation_factor,
            oscillation_period,
        )

        angle_per_frame = get_per_frame_rotation(current_rpm, fps)

        if blur_samples > 0:
            rotated = apply_motion_blur(frame, current_angle, angle_per_frame, blur_samples)
        else:
            M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
            rotated = warp(frame, M, w, h)

        if show_overlay:
            draw_rpm_overlay(rotated, current_rpm)
        out.write(rotated)

        current_angle = (current_angle + angle_per_frame) % 360.0
        frame_index += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a realistically accelerated rotating video.")

    parser.add_argument("source_video_path", type=str)
    parser.add_argument("--rpm", required=True, type=int, help="Target steady-state RPM after acceleration.")
    parser.add_argument("--out", required=True, type=str, help="Path to the output video file or directory.")
    parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Number of samples for motion blur. Use 0 for no blur.",
    )
    parser.add_argument(
        "--accel-duration",
        type=float,
        default=4.0,
        help="Seconds to accelerate from 0 RPM to target RPM.",
    )
    parser.add_argument(
        "--oscillation-factor",
        type=float,
        default=0.12,
        help="Fractional oscillation amplitude around the target RPM (e.g. 0.1 = +/-10%).",
    )
    parser.add_argument(
        "--oscillation-period",
        type=float,
        default=3.0,
        help="Seconds for one oscillation cycle around the target RPM.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable the RPM overlay on output frames.",
    )

    args = parser.parse_args()

    source_video_path: str = args.source_video_path
    rpm: int = args.rpm
    save_path: str = args.out
    blur_samples: int = args.blur
    accel_duration: float = args.accel_duration
    oscillation_factor: float = args.oscillation_factor
    oscillation_period: float = args.oscillation_period
    show_overlay: bool = not args.no_overlay

    if os.path.isdir(save_path):
        if not os.path.exists(save_path):
            raise FileNotFoundError("Destination folder doesn't exist")
        save_path = os.path.join(save_path, os.path.basename(source_video_path))
    else:
        output_parent = Path(save_path).parent
        if not output_parent.exists():
            raise FileNotFoundError("Destination folder doesn't exist")

    if not os.path.exists(source_video_path):
        raise FileNotFoundError(
            "Source file doesn't exist or path incorrect. Please use absolute paths"
        )

    rotate_video(
        source_video_path,
        save_path,
        rpm,
        blur_samples,
        accel_duration,
        oscillation_factor,
        oscillation_period,
    )
