import os
import math
import argparse
from pathlib import Path

import cv2
import numpy as np


def get_angular_velocity(rpm: int):
    angular_velocity = (
        2.0 * math.pi * rpm
    ) / 60  # return radians per second from revolutions per minute
    return angular_velocity


def get_per_frame_rotation(rpm: int, fps: float):
    per_frame_rotation = (360 * rpm) / (
        60 * fps
    )  # How much should each consecutive frame rotate
    return per_frame_rotation


def warp(frame, M, w, h):
    return cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def apply_motion_blur(frame, base_angle, angle_step, samples=5):
    """
    Simulates rotational motion blur by integrating multiple sub-rotations.
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    acc = np.zeros_like(frame, dtype=np.float32)

    for i in range(samples):
        a = base_angle + (i / samples) * angle_step
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        warped = warp(frame, M, w, h)
        acc += warped

    return (acc / samples).astype(np.uint8)


def rotate_video(
    src_path: str,
    destination_path: str,
    rpm: int,
    blur_samples: int,
):
    cap = cv2.VideoCapture(src_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(destination_path, fourcc, fps, (w, h))

    angle_per_frame = get_per_frame_rotation(rpm, fps)
    side = int(min(h, w) / np.sqrt(2))

    current_angle = 0.0
    center = (w // 2, h // 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if blur_samples:
            rotated = apply_motion_blur(
                frame,
                base_angle=current_angle,
                angle_step=angle_per_frame,
                samples=blur_samples,
            )
        else:
            M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
            rotated = warp(frame, M, w, h)

        out.write(rotated)

        current_angle += angle_per_frame
        current_angle %= 360  # clip at 360. Forgot to do this last time

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("source_video_path", type=str)
    parser.add_argument("--rpm", required=True, type=int)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--blur", required=True, type=int, default=0)

    args = parser.parse_args()

    source_video_path: str = args.source_video_path
    rpm: int = args.rpm
    save_path: str = args.out
    blur_samples: bool = args.blur

    if os.path.isdir(save_path):  # if output is a directory
        if not os.path.exists(save_path):  # if said directory doesn't exist
            raise FileNotFoundError("Destination folder doesn't exist")
        save_path = os.path.join(save_path, source_video_path.split(os.sep)[-1])
    else:
        if not os.path.exists(Path(save_path).parent):
            raise FileNotFoundError("Destination folder doesn't exist")

    if not os.path.exists(source_video_path):
        raise FileNotFoundError(
            "Source file doesn't exist or path incorrect. Please use absolute paths"
        )

    rotate_video(source_video_path, save_path, rpm, blur_samples)
