# ✍️ Air Writer — Virtual Writing System

Write in the air using just your index finger. No touch, no stylus, no surface.

## Requirements

```
Python 3.8+
opencv-python
mediapipe
numpy
```

Install with:
```bash
pip install opencv-python mediapipe numpy
```

## Run

```bash
python air_writer.py
# or
bash run.sh
```

---

## Gesture Reference

| Gesture | Action |
|---|---|
| ☝️ Index finger raised | **DRAW** — traces your finger path |
| ✌️ Index + Middle close together | **LIFT PEN** — ends current stroke |
| 🖐️ Open palm (hold ~0.5s) | **ERASE ALL** — clears the canvas |
| 🤌 Pinch on colour swatch | **CHANGE COLOUR** |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` | Quit |
| `C` | Clear canvas |
| `S` | Save drawing as PNG |
| `1–7` | Select colour (cyan, green, red, yellow, purple, orange, white) |
| `+` / `-` | Increase / decrease brush width |

---

## How It Works

1. **MediaPipe Hands** detects 21 hand landmarks at 30–60 fps.
2. The **index fingertip** (landmark #8) is tracked as the pen tip.
3. A **smoothing buffer** (rolling average of 6 points) eliminates jitter.
4. **Pen-lift detection**: when the middle fingertip (#12) comes close to the index tip, the stroke ends — mimicking lifting a pen.
5. **Palm-open erase**: all four finger tips above their bases for 12+ consecutive frames triggers a canvas clear.
6. **Canvas compositing**: ink strokes are drawn on a separate layer and alpha-blended over the webcam feed.

---

## Tips for Best Results

- Use in **good lighting** — front-lit, no harsh backlighting.
- Keep your hand **30–60 cm** from the camera.
- Write **slowly and clearly** for the smoothest strokes.
- Use **pen-lift gesture** between letters/words to avoid unwanted connecting lines.
- **Save** your drawing with `S` before clearing.

---

## File Output

Saved drawings are written to the working directory as:
```
airwrite_<unix_timestamp>.png
```
