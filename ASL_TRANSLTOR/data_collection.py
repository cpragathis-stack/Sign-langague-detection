import cv2
import mediapipe as mp
import csv
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

def create_hand_detector():
    base_options = BaseOptions(model_asset_path="hand_landmarker.task")
    options = HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        running_mode=VisionRunningMode.VIDEO
    )
    return HandLandmarker.create_from_options(options)

def extract_landmarks(detector, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        return [v for pt in lm for v in (pt.x, pt.y, pt.z)]
    return None

def draw_ui(frame, label, count, num_samples, collecting, countdown=0):
    h, w = frame.shape[:2]
    # Background bar
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    status = "COLLECTING..." if collecting else f"Get Ready: {countdown}s"
    color = (0, 255, 0) if collecting else (0, 165, 255)

    cv2.putText(frame, f"Sign: [{label}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"{count}/{num_samples}  {status}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Progress bar
    progress = int((count / num_samples) * w)
    cv2.rectangle(frame, (0, 58), (progress, 60), (0, 255, 0), -1)

    cv2.putText(frame, "ESC=quit  SPACE=skip letter", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def collect_static(label, num_samples=200, save_dir="data/static", countdown_secs=3):
    os.makedirs(save_dir, exist_ok=True)

    # Skip if already collected
    save_path = f"{save_dir}/{label}.csv"
    if os.path.exists(save_path):
        with open(save_path) as f:
            existing = sum(1 for _ in f)
        if existing >= num_samples:
            print(f"[SKIP] '{label}' already has {existing} samples.")
            return

    detector = create_hand_detector()
    cap = cv2.VideoCapture(0)
    samples = []
    start_time = time.time()
    aborted = False

    # --- Countdown phase ---
    countdown_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        elapsed = time.time() - countdown_start
        remaining = max(0, int(countdown_secs - elapsed) + 1)
        draw_ui(frame, label, 0, num_samples, collecting=False, countdown=remaining)
        cv2.imshow("ASL Data Collector", frame)
        key = cv2.waitKey(1)
        if key == 27:
            aborted = True
            break
        if key == 32:  # SPACE skips countdown early
            break
        if elapsed >= countdown_secs:
            break

    # --- Collection phase ---
    if not aborted:
        start_time = time.time()
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - start_time) * 1000)
            lm = extract_landmarks(detector, frame, timestamp_ms)

            if lm:
                samples.append(lm)

            draw_ui(frame, label, len(samples), num_samples, collecting=True)
            cv2.imshow("ASL Data Collector", frame)
            key = cv2.waitKey(1)
            if key == 27:   # ESC = quit everything
                aborted = True
                break
            if key == 32:   # SPACE = skip this letter
                break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # Save whatever was collected
    if samples:
        with open(save_path, "w", newline="") as f:
            csv.writer(f).writerows(samples)
        print(f"[SAVED] '{label}': {len(samples)} samples → {save_path}")
    else:
        print(f"[SKIP] '{label}': No samples collected.")

    return aborted  # signal to stop outer loop if ESC was pressed

if __name__ == "__main__":
    # Start with A-E, expand to full alphabet later
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    print("=== ASL Data Collector ===")
    print("SPACE = skip letter | ESC = quit all\n")

    for letter in letters:
        aborted = collect_static(letter, num_samples=200)
        if aborted:
            print("Aborted. Exiting.")
            break
        print(f"✓ Done '{letter}'. Resting 3 seconds...\n")
        time.sleep(3)

    print("=== Collection Complete ===")

