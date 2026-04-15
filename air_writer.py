import os
import sys
import math
import time
import urllib.request
import urllib.error
from collections import deque
 
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
 
 
# ─── MODEL AUTO-DOWNLOAD ─────────────────────────────────────────────────────
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
 
def ensure_model():
    """Download the hand-landmarker model if not present."""
    if os.path.exists(MODEL_FILENAME) and os.path.getsize(MODEL_FILENAME) > 100_000:
        return MODEL_FILENAME
    print(f"📥  Downloading hand landmark model → {MODEL_FILENAME}")
    print(f"    Source: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        size = os.path.getsize(MODEL_FILENAME)
        if size < 100_000:
            raise RuntimeError(f"Downloaded file too small ({size} bytes) — likely an error page.")
        print(f"✅  Model ready ({size // 1024} KB)")
    except Exception as e:
        print(f"\n❌  Could not download model: {e}")
        print("    Please download it manually from:")
        print(f"    {MODEL_URL}")
        print(f"    Save it as  '{MODEL_FILENAME}'  in the same folder as this script.")
        sys.exit(1)
    return MODEL_FILENAME
 
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
CANVAS_ALPHA      = 0.85
STROKE_WIDTH      = 4
SMOOTHING         = 6
MIN_DRAW_DIST     = 4
PALM_ERASE_FRAMES = 14
LIFT_THRESHOLD    = 0.06    # index-middle gap / frame width
 
PALETTE = [
    (0, 220, 255),   # cyan
    (0, 255, 120),   # green
    (80,  80, 255),  # red (BGR)
    (0,  200, 255),  # yellow (BGR)
    (200, 80, 255),  # purple
    (0,  140, 255),  # orange
    (255,255, 255),  # white
]
PAL_LABELS = ["CYN", "GRN", "RED", "YLW", "PRP", "ORG", "WHT"]
PAL_DISPLAY = [   # RGB for display circles (OpenCV is BGR)
    (0, 220, 255),
    (0, 255, 120),
    (80,  80, 255),
    (0,  200, 255),
    (200, 80, 255),
    (0,  140, 255),
    (255,255, 255),
]
 
 
# ─── HAND LANDMARK INDICES ───────────────────────────────────────────────────
IDX_TIP  = 8
IDX_MID  = 6
MID_TIP  = 12
THUMB_TIP = 4
FINGER_TIPS  = [8, 12, 16, 20]
FINGER_BASES = [6, 10, 14, 18]
 
 
# ─── UTILS ───────────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
 
 
def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)
 
 
def smooth(buf):
    if not buf:
        return None
    xs = [p[0] for p in buf]
    ys = [p[1] for p in buf]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
 
 
def palm_open(lms, w, h):
    extended = 0
    for tip_i, base_i in zip(FINGER_TIPS, FINGER_BASES):
        if lms[tip_i].y < lms[base_i].y:   # tip higher (smaller y) than base
            extended += 1
    return extended >= 4
 
 
def pen_lifted(lms, w, h):
    a = lm_px(lms[IDX_TIP], w, h)
    b = lm_px(lms[MID_TIP],  w, h)
    return dist(a, b) / w < LIFT_THRESHOLD
 
 
# ─── HUD ─────────────────────────────────────────────────────────────────────
def draw_hud(frame, mode, color_idx, palm_ctr, stroke_w, fps):
    h, w = frame.shape[:2]
 
    # top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
 
    cv2.putText(frame, "AIR WRITER",
                (14, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
 
    badge_col = {
        "DRAW":  (0, 200, 80),
        "LIFT":  (80, 120, 220),
        "ERASE": (60, 60, 200),
    }.get(mode, (160, 160, 160))
 
    cv2.putText(frame, f"[ {mode} ]",
                (w // 2 - 55, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, badge_col, 2, cv2.LINE_AA)
 
    cv2.putText(frame, f"{fps:.0f}fps",
                (w - 80, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
 
    # colour palette (bottom left)
    py = h - 52
    for i, c in enumerate(PALETTE):
        cx = 14 + i * 44 + 16
        filled = i == color_idx
        cv2.circle(frame, (cx, py + 16), 15, c, -1 if filled else 2)
        if filled:
            cv2.circle(frame, (cx, py + 16), 18, (255, 255, 255), 2)
 
    # stroke width indicator
    sw_x = 14 + len(PALETTE) * 44 + 8
    cv2.putText(frame, f"W:{stroke_w}",
                (sw_x, py + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
 
    # erase progress bar
    if palm_ctr > 0:
        ratio  = min(palm_ctr / PALM_ERASE_FRAMES, 1.0)
        bw     = int(220 * ratio)
        bx     = w // 2 - 110
        by     = h - 28
        cv2.rectangle(frame, (bx, by), (bx + 220, by + 16), (35, 35, 35), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw,  by + 16), (50, 50, 200), -1)
        cv2.putText(frame, "OPEN PALM  →  ERASE",
                    (bx + 10, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA)
 
    # shortcuts hint
    cv2.putText(frame,
                "[Q] Quit  [C] Clear  [S] Save  [+/-] Width  [1-7] Color",
                (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
 
 
def draw_cursor(frame, pt, mode, color, sw):
    if pt is None:
        return
    if mode == "DRAW":
        cv2.circle(frame, pt, sw + 5,  color, -1)
        cv2.circle(frame, pt, sw + 9,  color,  1)
    else:
        cv2.circle(frame, pt, 10, (90, 90, 90), 2)
        cv2.line(frame,  (pt[0]-7, pt[1]), (pt[0]+7, pt[1]), (90, 90, 90), 1)
        cv2.line(frame,  (pt[0], pt[1]-7), (pt[0], pt[1]+7), (90, 90, 90), 1)
 
 
def palette_hit(px, py, fh, fw):
    """Return palette index if tap lands on swatch, else -1."""
    row_y = fh - 52 + 16
    for i in range(len(PALETTE)):
        cx = 14 + i * 44 + 16
        if dist((px, py), (cx, row_y)) < 18:
            return i
    sw_x = 14 + len(PALETTE) * 44 + 8
    if sw_x - 4 < px < sw_x + 50 and fh - 52 < py < fh:
        return "width"
    return -1
 
 
# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    model_path = ensure_model()
 
    # Build HandLandmarker (Tasks API)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.55,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
 
    ret, frame = cap.read()
    if not ret:
        print("❌  Cannot open webcam.")
        landmarker.close()
        return
 
    fh, fw = frame.shape[:2]
    canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
 
    color_idx    = 0
    stroke_width = STROKE_WIDTH
    mode         = "LIFT"
    prev_pt      = None
    last_pt      = None
    palm_ctr     = 0
    pinch_cool   = 0
    smooth_buf   = deque(maxlen=SMOOTHING)
    cur_pts      = []
 
    fps_buf = deque([30.0] * 10, maxlen=10)
    t_prev  = time.time()
    ts_ms   = 0          # timestamp for Tasks API (must be monotonically increasing)
 
    cv2.namedWindow("Air Writer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Writer", fw, fh)
 
    print("\n✍️   Air Writer running!")
    print("   ☝  Index finger up            → DRAW")
    print("   ✌  Index + Middle close       → LIFT pen")
    print("   🖐  Open palm (hold ~0.5s)     → ERASE all")
    print("   Keyboard: Q quit | C clear | S save | 1-7 colour | +/- width\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame = cv2.flip(frame, 1)
 
        # ── FPS ──────────────────────────────────────────────────────────────
        t_now = time.time()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-9))
        t_prev = t_now
        fps = sum(fps_buf) / len(fps_buf)
 
        # ── Hand detection (Tasks API) ────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms += int(1000 / max(fps, 1))
        result = landmarker.detect_for_video(mp_img, ts_ms)
 
        hand_found = bool(result.hand_landmarks)
 
        if hand_found:
            lms = result.hand_landmarks[0]   # list of NormalizedLandmark
 
            # Draw skeleton manually (Tasks API has no draw_landmarks helper)
            CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
            for conn in CONNECTIONS:
                a = lm_px(lms[conn.start], fw, fh)
                b = lm_px(lms[conn.end],   fw, fh)
                cv2.line(frame, a, b, (60, 60, 60), 1, cv2.LINE_AA)
            for lm in lms:
                cv2.circle(frame, lm_px(lm, fw, fh), 3, (120, 120, 120), -1)
 
            index_tip = lm_px(lms[IDX_TIP], fw, fh)
            smooth_buf.append(index_tip)
            sp = smooth(smooth_buf)
 
            is_palm = palm_open(lms, fw, fh)
            is_up   = pen_lifted(lms, fw, fh)
 
            # ── gesture → mode ────────────────────────────────────────────────
            if is_palm:
                mode = "ERASE"
                palm_ctr += 1
                if palm_ctr >= PALM_ERASE_FRAMES:
                    canvas[:] = 0
                    cur_pts.clear()
                    palm_ctr = 0
            elif is_up:
                mode = "LIFT"
                palm_ctr = 0
                if cur_pts:
                    cur_pts.clear()
                prev_pt = None
            else:
                mode = "DRAW"
                palm_ctr = 0
 
            # ── pinch → palette / width selection ─────────────────────────────
            thumb  = lm_px(lms[THUMB_TIP], fw, fh)
            if dist(index_tip, thumb) < 30 and pinch_cool == 0:
                hit = palette_hit(index_tip[0], index_tip[1], fh, fw)
                if hit == "width":
                    stroke_width = max(2, min(20, stroke_width + 2))
                elif isinstance(hit, int) and hit >= 0:
                    color_idx = hit
                pinch_cool = 20
            if pinch_cool > 0:
                pinch_cool -= 1
 
            # ── draw ──────────────────────────────────────────────────────────
            if mode == "DRAW" and sp:
                if prev_pt and dist(sp, prev_pt) > MIN_DRAW_DIST:
                    cv2.line(canvas, prev_pt, sp,
                             PALETTE[color_idx], stroke_width, cv2.LINE_AA)
                    cur_pts.append(sp)
                elif not prev_pt:
                    cur_pts.append(sp)
                prev_pt = sp
 
            last_pt = sp
 
        else:
            smooth_buf.clear()
            palm_ctr = max(0, palm_ctr - 1)
            mode = "LIFT"
            if cur_pts:
                cur_pts.clear()
            prev_pt = None
 
        # ── composite canvas over webcam ──────────────────────────────────────
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        mask3  = cv2.merge([mask, mask, mask])
        blended = cv2.addWeighted(frame, 1.0, canvas, CANVAS_ALPHA, 0)
        display = np.where(mask3 > 0, blended, frame)
 
        # ── HUD & cursor ──────────────────────────────────────────────────────
        draw_hud(display, mode, color_idx, palm_ctr, stroke_width, fps)
        if hand_found:
            draw_cursor(display, last_pt, mode, PALETTE[color_idx], stroke_width)
 
        cv2.imshow("Air Writer", display)
 
        # ── keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            canvas[:] = 0
            cur_pts.clear()
        elif key == ord('s'):
            fname = f"airwrite_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"💾  Saved → {fname}")
        elif key in (ord('+'), ord('=')):
            stroke_width = min(20, stroke_width + 1)
        elif key == ord('-'):
            stroke_width = max(2, stroke_width - 1)
        elif ord('1') <= key <= ord('7'):
            color_idx = key - ord('1')
 
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("✅  Air Writer closed.")
 
 
if __name__ == "__main__":
    main()