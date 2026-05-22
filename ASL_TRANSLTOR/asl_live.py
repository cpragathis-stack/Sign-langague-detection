import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os

# ── Load trained model ────────────────────────────────────────────────────────
MODEL_PATH = "model/asl_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Run translator.py first to train the model.")
    print(f"   Expected at: {MODEL_PATH}")
    exit(1)

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
    model  = data["model"]
    labels = data["labels"]

print(f"✅ Model loaded — knows {len(labels)} signs: {labels}")

# ── MediaPipe setup (same as your test_camera.py) ────────────────────────────
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions           = mp.tasks.BaseOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

base_options = BaseOptions(model_asset_path="hand_landmarker.task")
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    running_mode=VisionRunningMode.VIDEO
)
detector = HandLandmarker.create_from_options(options)

# ── Feature extraction (matches your data_collection.py format) ──────────────
def extract_features(hand_landmarks):
    """
    Flatten all 21 landmarks into [x0,y0,z0, x1,y1,z1 ... x20,y20,z20]
    — same 63-feature vector your data_collection.py saves to CSV.
    """
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features).reshape(1, -1)

# ── State ─────────────────────────────────────────────────────────────────────
sentence          = ""
current_letter    = ""
last_letter       = ""
letter_hold_t     = 0.0
HOLD_SECONDS      = 1.2       # hold sign this long to confirm it
REPEAT_GAP        = 2.0       # seconds before same letter can repeat
last_added_t      = 0.0
last_added_letter = ""
confidence_val    = 0.0

# ── Drawing helpers ───────────────────────────────────────────────────────────
FONT   = cv2.FONT_HERSHEY_DUPLEX
CYAN   = (0, 230, 200)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0, 220,  80)
RED    = (0,  60, 220)
AMBER  = (0, 180, 255)
DGRAY  = (30,  30,  30)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def draw_hand(frame, hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame, (x, y), 5, CYAN, -1)
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], WHITE, 2)

def draw_letter_bubble(frame, letter, progress, conf, w):
    """Top-right: big predicted letter + confidence + hold ring."""
    bx, by, bw, bh = w - 160, 10, 140, 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), DGRAY, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), CYAN, 1)

    # progress ring
    if progress > 0:
        center = (bx + bw // 2, by + bh // 2)
        cv2.ellipse(frame, center, (58, 58), -90, 0,
                    int(360 * progress), CYAN, 4)

    # letter
    scale = 3.0 if len(letter) == 1 else 1.2
    (tw, th), _ = cv2.getTextSize(letter, FONT, scale, 3)
    tx = bx + (bw - tw) // 2
    ty = by + (bh + th) // 2 - 8
    cv2.putText(frame, letter, (tx, ty), FONT, scale, WHITE, 3, cv2.LINE_AA)

    # confidence bar
    bar_x, bar_y = bx + 8, by + bh + 10
    bar_w = int((bw - 16) * conf)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bw - 16, bar_y + 8), DGRAY, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), GREEN, -1)
    cv2.putText(frame, f"{int(conf*100)}%", (bar_x, bar_y + 22),
                FONT, 0.5, AMBER, 1, cv2.LINE_AA)

def draw_subtitle(frame, text, h, w):
    """Bottom subtitle bar."""
    bar_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    display = text[-52:] if len(text) > 52 else text
    cv2.putText(frame, display if display else "[ show a sign… ]",
                (18, h - 20), FONT, 1.1, WHITE, 2, cv2.LINE_AA)

def draw_hud(frame, fps, h):
    cv2.putText(frame, f"ASL TRANSLATOR", (12, 35),
                FONT, 0.9, CYAN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}", (12, 62),
                FONT, 0.55, AMBER, 1, cv2.LINE_AA)
    cv2.putText(frame, "Hold sign 1s=add | SPACE=space | BKSP=delete | C=clear | ESC=quit",
                (12, h - 82), FONT, 0.42, AMBER, 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start_time = time.time()
prev_t     = time.time()

print("✅ ASL Translator running — press ESC to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame        = cv2.flip(frame, 1)
    h, w         = frame.shape[:2]
    now          = time.time()
    fps          = 1.0 / max(now - prev_t, 1e-6)
    prev_t       = now
    timestamp_ms = int((now - start_time) * 1000)

    # ── Detection ─────────────────────────────────────────────────────────────
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect_for_video(mp_image, timestamp_ms)

    detected   = ""
    confidence_val = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        draw_hand(frame, hand, w, h)

        # Predict
        features = extract_features(hand)
        pred     = model.predict(features)[0]
        proba    = model.predict_proba(features)[0]
        conf     = float(np.max(proba))

        if conf >= 0.60:               # only accept confident predictions
            detected       = str(pred).upper()
            confidence_val = conf

    # ── Hold-to-confirm logic ─────────────────────────────────────────────────
    progress = 0.0
    if detected:
        if detected == last_letter:
            elapsed  = now - letter_hold_t
            progress = min(elapsed / HOLD_SECONDS, 1.0)
            if elapsed >= HOLD_SECONDS:
                gap_ok = (detected != last_added_letter or
                          (now - last_added_t) >= REPEAT_GAP)
                if gap_ok:
                    sentence         += detected
                    last_added_letter = detected
                    last_added_t      = now
                letter_hold_t = now    # reset timer
        else:
            last_letter   = detected
            letter_hold_t = now
    else:
        last_letter = ""

    # ── Draw UI ───────────────────────────────────────────────────────────────
    if detected:
        draw_letter_bubble(frame, detected, progress, confidence_val, w)

    draw_subtitle(frame, sentence, h, w)
    draw_hud(frame, fps, h)

    cv2.imshow("ASL Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:       # ESC
        break
    elif key == 32:     # SPACE
        sentence += " "
    elif key == 8:      # BACKSPACE
        sentence = sentence[:-1]
    elif key == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()
detector.close()
print(f"\nFinal sentence: {sentence}")
