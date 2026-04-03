import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.ops
from ultralytics import YOLO
import kornia.geometry as K
import kornia.feature as KF

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

import onnx


def expected_angle_step(rpm, fps):
    return (360.0 * rpm) / (60.0 * fps)


def build_trt_engine(onnx_path, engine_path, precision='fp16'):
    if not TRT_AVAILABLE:
        raise RuntimeError('TensorRT not installed')

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            raise RuntimeError('Failed to parse ONNX model')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: int8 calibration is required for real use; here w/ dynamic fallback.

    # Build optimization profile for dynamic input shape
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 3, 320, 320), (1, 3, 640, 640), (1, 3, 960, 960))
    config.add_optimization_profile(profile)

    try:
        engine = builder.build_engine(network, config)
    except AttributeError:
        runtime = trt.Runtime(logger)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError('TensorRT engine build failed')
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    if engine is None:
        raise RuntimeError('TensorRT engine build failed')

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    return engine


def load_trt_engine(engine_path):
    if not TRT_AVAILABLE:
        raise RuntimeError('TensorRT not installed')
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(f.read())


def trt_infer(engine, input_array):
    """
    Run inference on TensorRT engine (YOLO exported).
    Args:
        engine: TensorRT engine
        input_array: (H, W, 3) uint8 BGR frame
    Returns:
        output: (1, num_outputs, 85) YOLO tensor format
    """
    context = engine.create_execution_context()
    
    # Ensure input is contiguous and float32
    input_array = np.ascontiguousarray(input_array, dtype=np.float32)
    h_in, w_in = input_array.shape[:2]
    
    # Normalize and convert to CHW format (YOLO format)
    input_data = input_array / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
    input_data = np.expand_dims(input_data, 0)  # Add batch: (1, 3, H, W)
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    
    # Get tensor names
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # Set input shape for dynamic input
    context.set_input_shape(input_name, input_data.shape)
    
    # CUDA memory allocation
    stream = cuda.Stream()
    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod_async(d_input, input_data, stream)
    
    # YOLO output format: (batch, num_detections, 85)
    output_shape = (1, 25200, 85)
    output_data = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)
    
    # Set tensor addresses for v3 API
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Execute inference with async v3 API
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()
    
    return output_data  # Raw YOLO tensor (1, 25200, 85)


def parse_yolo_output(raw_output, conf_thresh=0.25, iou_thresh=0.45):
    """
    Parse raw YOLO tensor output (1, 25200, 85) into detections.
    Uses torchvision NMS for post-processing.
    
    Args:
        raw_output: (1, 25200, 85) array where last 85 = [x,y,w,h,conf,c1,...,c80]
        conf_thresh: confidence threshold
        iou_thresh: NMS IOU threshold
    
    Returns:
        List of detections: [(x1, y1, x2, y2, conf, cls), ...]
    """
    detections = []
    output = raw_output[0]  # (25200, 85)
    
    # Extract confidence and class scores
    conf = output[:, 4]
    class_scores = output[:, 5:85]
    max_class_conf, class_ids = np.max(class_scores, axis=1), np.argmax(class_scores, axis=1)
    
    # Filter by confidence threshold
    mask = (conf * max_class_conf) > conf_thresh
    filtered_output = output[mask]
    filtered_class_ids = class_ids[mask]
    
    if len(filtered_output) == 0:
        return detections
    
    # Convert xywh to xyxy
    x_centers = filtered_output[:, 0]
    y_centers = filtered_output[:, 1]
    widths = filtered_output[:, 2]
    heights = filtered_output[:, 3]
    
    x1 = x_centers - widths / 2
    y1 = y_centers - heights / 2
    x2 = x_centers + widths / 2
    y2 = y_centers + heights / 2
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = conf[mask] * max_class_conf[mask]
    
    # Apply NMS per class
    keep_indices = []
    for cls_id in np.unique(filtered_class_ids):
        cls_mask = filtered_class_ids == cls_id
        cls_boxes = torch.from_numpy(boxes[cls_mask]).float()
        cls_scores = torch.from_numpy(scores[cls_mask]).float()
        cls_keep = torchvision.ops.nms(cls_boxes, cls_scores, iou_thresh)
        keep_indices.extend(np.where(cls_mask)[0][cls_keep.numpy()])
    
    keep_indices = sorted(set(keep_indices))
    
    # Build detections list
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        conf = scores[idx]
        cls = filtered_class_ids[idx]
        detections.append((x1, y1, x2, y2, conf, cls))
    
    return detections


class MockBox:
    """Mock YOLO box object for TensorRT output compatibility."""
    def __init__(self, x1, y1, x2, y2, conf, cls):
        self.xyxy = np.array([[x1, y1, x2, y2]])
        self.conf = np.array([conf])
        self.cls = np.array([cls])


class MockResults:
    """Mock YOLO Results object for TensorRT output compatibility."""
    def __init__(self, detections, names):
        self.boxes = [MockBox(x1, y1, x2, y2, conf, cls) for x1, y1, x2, y2, conf, cls in detections]
        self.names = names


def export_yolo_onnx(yolo_model_path, onnx_path):
    model = YOLO(yolo_model_path)
    model.export(format='onnx', imgsz=640, dynamic=True)
    # ultralytics exports to yolov8.onnx; move to desired location
    if os.path.exists('yolov8.onnx') and not os.path.exists(onnx_path):
        os.rename('yolov8.onnx', onnx_path)


def estimate_rotation_ecc(prev_gray, curr_gray):
    # Removed: Use predicted rotation only for speed
    return 0.0


def estimate_rotation_ecc_fast(prev_gray, curr_gray, max_dim=480):
    # Removed: Simplified to predicted
    return 0.0


def warp_affine(frame, M, dsize):
    """
    Warp frame using affine transformation.
    Uses Kornia GPU path if CUDA available, falls back to OpenCV.
    Args:
        frame: (H, W, 3) uint8 BGR image
        M: (2, 3) transformation matrix
        dsize: (width, height) output size
    Returns:
        Warped frame with same dtype and size as input
    """
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for Kornia path")
        
        # Input validation
        if frame.shape[2] != 3:
            raise ValueError(f"Expected 3-channel frame, got {frame.shape[2]}")
        
        h_in, w_in = frame.shape[:2]
        w_out, h_out = dsize
        
        # Convert frame to tensor [0, 1]
        frame_t = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        frame_t = frame_t / 255.0
        frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        frame_t = frame_t.cuda()
        
        # Prepare transformation matrix (2x3, no batch for Kornia)
        M_t = torch.from_numpy(M, dtype=torch.float32).cuda()
        # Kornia expects (1, 2, 3) with batch dimension
        M_t = M_t.unsqueeze(0)
        
        # Warp affine
        out = K.warp_affine(frame_t, M_t, dsize=dsize, align_corners=False, padding_mode='reflection')
        
        # Convert back to uint8 BGR
        out = out[0].permute(1, 2, 0)  # (H, W, 3)
        out = out.cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        
        # Ensure contiguous and correct size
        if out.shape != (h_out, w_out, 3):
            raise ValueError(f"Output shape {out.shape} != expected {(h_out, w_out, 3)}")
        
        return np.ascontiguousarray(out)
    except Exception as e:
        # Fallback to OpenCV
        return cv2.warpAffine(frame, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def run_viewer(video_path: str, rpm: int, precision: str = 'fp16'):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model = YOLO("yolo26n.pt")
    draw_detections = True

    print(f"PyTorch CUDA enabled: {torch.cuda.is_available()}")
    print(f"Requested TensorRT precision: {precision}")

    onnx_path = "yolo26n.onnx"
    engine_path = f"yolo26n_{precision}.engine"
    trt_engine = None

    if TRT_AVAILABLE:
        if not os.path.exists(onnx_path):
            export_yolo_onnx("yolo26n.pt", onnx_path)

        if not os.path.exists(engine_path):
            print(f"TensorRT engine not found, building: {engine_path}")
            build_trt_engine(onnx_path, engine_path, precision=precision)

        trt_engine = load_trt_engine(engine_path)
        print(f"Loaded TensorRT engine: {engine_path}")
    else:
        print("TensorRT not available; running ultralytics model.predict path")

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
        
        # Try TensorRT path first if available
        if trt_engine is not None:
            try:
                raw_trt_output = trt_infer(trt_engine, infer_frame)
                detections = parse_yolo_output(raw_trt_output, conf_thresh=0.25, iou_thresh=0.45)
                results = MockResults(detections, model.names)
            except Exception as e:
                # Fallback to YOLO if TRT fails
                results = model.predict(infer_frame, half=True, verbose=False)[0]
        else:
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
    parser.add_argument("--precision", choices=["fp16", "int8", "fp32"], default="fp16")

    args = parser.parse_args()

    run_viewer(args.video_path, args.rpm, precision=args.precision)