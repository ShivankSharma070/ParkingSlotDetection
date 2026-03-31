import cv2
import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] 'ultralytics' package is required but not installed.")
    print("Please install requirements by running: pip install ultralytics\n")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Local YOLO Inference Script")
    parser.add_argument("--model", type=str, required=True, help="Path to your locally trained .pt file (e.g., best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--iou", type=float, default=0.30, help="IoU Overlap threshold (default: 0.30)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for inference. Default 1280 for dense lots.")
    parser.add_argument("--skip-frames", type=int, default=5, help="Run inference every N frames to speed up videos (default 5).")
    return parser.parse_args()

def run_inference(model_path, source_path, conf_thresh, iou_thresh, imgsz_val, skip_frames):
    if not Path(model_path).exists():
        print(f"\n[ERROR] Model file not found: {model_path}\n")
        return
        
    if not Path(source_path).exists():
        print(f"\n[ERROR] Source file not found: {source_path}\n")
        return

    print(f"\n[INFO] Loading locally trained PyTorch model: {model_path}...")
    model = YOLO(model_path)
    print(f"[INFO] Model loaded successfully.")

    COLORS = {
        0: (68, 68, 239),  # BGR for Red (#ef4444) 
        1: (94, 197, 34)   # BGR for Green (#22c55e)
    }

    print(f"[INFO] Opening source: {source_path}...")
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open source video/image.\n")
        return
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video = frame_count > 1

    print("\n==============================================")
    print(f"Running Inference: Conf ≥ {conf_thresh:.2f} | IoU ≤ {iou_thresh:.2f} | Imgsz = {imgsz_val} | Skip = {skip_frames}")
    if is_video:
        print("Video mode active. Press 'q' to stop window.")
    else:
        print("Image mode active. Press any key to close the window.")
    print("==============================================\n")

    window_name = "Local YOLO Parking Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_id = 0
    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if is_video:
                print("\n[INFO] End of video reached.")
            break

        # Only run expensive YOLO inference every Nth frame
        if frame_id % skip_frames == 0:
            results = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, imgsz=imgsz_val, verbose=False)
            result = results[0]
            last_boxes = result.boxes

        frame_id += 1
        
        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            color = COLORS.get(cls_id, (255, 255, 255))
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow(window_name, frame)

        if is_video:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User requested stop.")
                break
        else:
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference complete.\n")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.source, args.conf, args.iou, args.imgsz, getattr(args, 'skip_frames', 5))
