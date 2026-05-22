import cv2
import mediapipe as mp

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create detector
base_options = BaseOptions(model_asset_path="hand_landmarker.task")
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    running_mode=VisionRunningMode.VIDEO
)
detector = HandLandmarker.create_from_options(options)

import time
cap = cv2.VideoCapture(0)
start_time = time.time()

print("Camera open! Show your hand. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    timestamp_ms = int((time.time() - start_time) * 1000)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        h, w = frame.shape[:2]
        for hand in result.hand_landmarks:
            pts = []
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                pts.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        cv2.putText(frame, "Hand Detected!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Done!")