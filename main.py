from ultralytics import YOLO
import cv2
import time
import supervision as sv
from collections import defaultdict, deque
import numpy as np

# LOAD MODEL
model = YOLO("yolov8n.pt")

# VIDEO INPUT
video_path = "input/video1.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# OUTPUT VIDEO
output_path = "output_video1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

tracker = sv.ByteTrack()

track_history = defaultdict(
    lambda: deque(maxlen=30)
)
heatmap = np.zeros(
    (height, width),
    dtype=np.float32
)

box_annotator = sv.BoxAnnotator(
    thickness=2
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1
)

orb = cv2.ORB_create(2000)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

ret, prev_frame = cap.read()

if not ret:
    print("Error reading video")
    exit()

prev_gray = cv2.cvtColor(
    prev_frame,
    cv2.COLOR_BGR2GRAY
)

prev_time = 0
unique_ids = set()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    kp1, des1 = orb.detectAndCompute(
        prev_gray,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray,
        None
    )

    stabilized_frame = frame.copy()

    if des1 is not None and des2 is not None:

        matches = bf.match(des1, des2)

        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

        if len(matches) > 10:

            src_pts = np.float32([
                kp1[m.queryIdx].pt
                for m in matches
            ]).reshape(-1, 1, 2)

            dst_pts = np.float32([
                kp2[m.trainIdx].pt
                for m in matches
            ]).reshape(-1, 1, 2)

            matrix, _ = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts
            )

            if matrix is not None:

                stabilized_frame = cv2.warpAffine(
                    frame,
                    matrix,
                    (width, height)
                )

    prev_gray = gray.copy()

    tiles = []

    half_w = width // 2
    half_h = height // 2

    tiles.append((0, 0, stabilized_frame[0:half_h, 0:half_w]))
    tiles.append((half_w, 0, stabilized_frame[0:half_h, half_w:width]))
    tiles.append((0, half_h, stabilized_frame[half_h:height, 0:half_w]))
    tiles.append((half_w, half_h, stabilized_frame[half_h:height, half_w:width]))

    all_xyxy = []
    all_conf = []
    all_class_id = []

    for offset_x, offset_y, tile in tiles:

        results = model(
            tile,
            imgsz=1280,
            conf=0.2
        )[0]

        detections = sv.Detections.from_ultralytics(
            results
        )

        for i in range(len(detections)):

            class_id = detections.class_id[i]

            if class_id == 0:

                x1, y1, x2, y2 = detections.xyxy[i]

                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y

                all_xyxy.append([x1, y1, x2, y2])
                all_conf.append(detections.confidence[i])
                all_class_id.append(class_id)

    if len(all_xyxy) > 0:

        detections = sv.Detections(
            xyxy=np.array(all_xyxy),
            confidence=np.array(all_conf),
            class_id=np.array(all_class_id)
        )

    else:

        detections = sv.Detections.empty()

    tracked_detections = tracker.update_with_detections(
        detections
    )


    for tracker_id in tracked_detections.tracker_id:
        unique_ids.add(tracker_id)

    labels = [
        f"PERSON #{tracker_id}"
        for tracker_id
        in tracked_detections.tracker_id
    ]


    annotated_frame = box_annotator.annotate(
        scene=stabilized_frame.copy(),
        detections=tracked_detections
    )
  
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=tracked_detections,
        labels=labels
    )
    for i in range(len(tracked_detections)):

        tracker_id = tracked_detections.tracker_id[i]

        x1, y1, x2, y2 = tracked_detections.xyxy[i]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        track_history[tracker_id].append(
            (center_x, center_y)
        )

        points = track_history[tracker_id]

        for j in range(1, len(points)):

            thickness = int(
                np.sqrt(30 / float(j + 1)) * 2
            )

            cv2.line(
                annotated_frame,
                points[j - 1],
                points[j],
                (0, 255, 255),
                thickness
            )

        cv2.circle(
            annotated_frame,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )
        cv2.circle(
            heatmap,
            (center_x, center_y),
            20,
            1,
            -1
        )
    heatmap_blur = cv2.GaussianBlur(
        heatmap,
        (51, 51),
        0
    )

    heatmap_normalized = cv2.normalize(
        heatmap_blur,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(
        heatmap_normalized,
        cv2.COLORMAP_JET
    )

    annotated_frame = cv2.addWeighted(
        annotated_frame,
        0.75,
        heatmap_color,
        0.25,
        0
    )

    current_time = time.time()

    fps_text = 1 / (current_time - prev_time)

    prev_time = current_time


    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps_text)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Current Persons: {len(tracked_detections)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Total Unique Persons: {len(unique_ids)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Drone Surveillance Analytics",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    out.write(annotated_frame)

    cv2.imshow(
        "Analytics MOT System",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()

print(
    f"Analytics tracking video saved at: {output_path}"
)
