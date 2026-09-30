"""Real-time monkey detection from a webcam or video file using a trained YOLOv8 model.

Usage:
    python detect_monkey.py                          # default webcam
    python detect_monkey.py --source video.mp4       # video file
    python detect_monkey.py --conf 0.4               # stricter confidence threshold

Press 'q' to quit.
"""
import argparse

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 monkey detection")
    parser.add_argument("--weights", default="weights/best.pt", help="path to trained weights")
    parser.add_argument("--source", default="0", help="webcam index (0, 1, ...) or path to a video file")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source '{args.source}'.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("End of stream or failed to read a frame.")
            break

        results = model(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        count = len(results[0].boxes)
        cv2.putText(annotated, f"Monkeys detected: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Monkey Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
