import cv2
import numpy as np
import torch
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

video_path = 'video.mp4'  
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}, Frame Count: {frame_count}, Duration: {duration} seconds")

quadrants = {
    1: ((frame_width//2, frame_height//2), (frame_width, frame_height)),
    2: ((0, frame_height//2), (frame_width//2, frame_height)),
    3: ((0, 0), (frame_width//2, frame_height//2)),
    4: ((frame_width//2, 0), (frame_width, frame_height//2)),
}

color_ranges = {
    'Yellow': [(25, 100, 100), (35, 255, 255)],  
    'Green': [(35, 50, 50), (85, 255, 255)],    
    'Red': [(0, 100, 100), (10, 255, 255)],     
    'White': [(0, 0, 200), (180, 20, 255)]      
}

events = []

def log_event(time, quadrant, color, event_type):
    events.append((time, quadrant, color, event_type))

def detect_balls(frame):
    results = model(frame)
    return results

def draw_quadrants(frame):
    for q, ((x1, y1), (x2, y2)) in quadrants.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Q{q}', (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def annotate_frame(frame, annotations):
    for ann in annotations:
        x, y, color, event, x1, y1, x2, y2 = ann
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (x, y), 10, color, -1)
        cv2.putText(frame, event, (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_color_name(hsv_pixel):
    hsv_pixel = np.uint8([[hsv_pixel]])  
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        if cv2.inRange(hsv_pixel, lower, upper).any():
            return color_name
    return 'Unknown'

output_video_path = 'output_video.mp4'
output_text_path = 'events.txt'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
ball_positions = {}  

with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        draw_quadrants(frame)
        results = detect_balls(frame)
        annotations = []

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for result in results.xyxy[0]:  
            x1, y1, x2, y2, conf, cls = result
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cx = max(0, min(cx, frame_width - 1))
            cy = max(0, min(cy, frame_height - 1))

            hsv_pixel = hsv_frame[cy, cx]
            color_name = get_color_name(hsv_pixel)
            color = (0, 0, 255)  

            if color_name == 'Yellow':
                color = (0, 255, 255)
            elif color_name == 'Green':
                color = (0, 255, 0)
            elif color_name == 'Red':
                color = (0, 0, 255)
            elif color_name == 'White':
                color = (255, 255, 255)

            quadrant = None
            for q, ((qx1, qy1), (qx2, qy2)) in quadrants.items():
                if qx1 <= cx <= qx2 and qy1 <= cy <= qy2:
                    quadrant = q
                    break

            if quadrant is not None:
                if color not in ball_positions:
                    ball_positions[color] = quadrant
                    log_event(timestamp, quadrant, color_name, "Entry")
                    annotations.append((cx, cy, color, "Entry", int(x1), int(y1), int(x2), int(y2)))
                elif ball_positions[color] != quadrant:
                    log_event(timestamp, ball_positions[color], color_name, "Exit")
                    annotations.append((cx, cy, color, "Exit", int(x1), int(y1), int(x2), int(y2)))
                    ball_positions[color] = quadrant
                    log_event(timestamp, quadrant, color_name, "Entry")
                    annotations.append((cx, cy, color, "Entry", int(x1), int(y1), int(x2), int(y2)))
                else:
                    annotations.append((cx, cy, color, "Tracking", int(x1), int(y1), int(x2), int(y2)))

        annotate_frame(frame, annotations)
        out.write(frame)
        frame_idx += 1
        pbar.update(1)

cap.release()
out.release()

with open(output_text_path, 'w') as f:
    for event in events:
        f.write(','.join(map(str, event)) + '\n')

print("Processing complete. Output saved to:", output_video_path, "and", output_text_path)

