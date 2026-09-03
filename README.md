# Air Canvas 🎨

Draw in the air with your index finger, tracked live via your webcam using MediaPipe's HandLandmarker.

Built with Next.js 16, React 19, MediaPipe Tasks Vision, and Tailwind CSS 4.

## Gestures

| Gesture | Action |
|---|---|
| 🤏 Pinch (thumb + index) | Draw |
| 🖐 Open palm | Erase |
| ✌️ Peace sign | Undo last stroke |
| ✊ Fist | Clear everything |

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Download the hand-tracking model (not bundled in this zip — ~7.5MB binary):
   - Grab it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   - Save it as `public/models/hand_landmarker.task`
   - Delete `public/models/PLACE_MODEL_HERE.txt`

3. Run the dev server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3000, allow camera access, and start drawing.

## How it works

- **Three stacked canvases** avoid the classic bug where redrawing the video feed each frame wipes your strokes:
  1. `videoCanvasRef` — repainted every frame with the mirrored webcam image.
  2. `drawCanvasRef` — only ever appended to; strokes persist until you hit Clear.
  3. `cursorCanvasRef` — cleared and redrawn every frame for the fingertip indicator, so it never leaves ghost trails and never touches your actual drawing.
- **Gesture detection**: each of the four non-thumb fingers is classified extended/curled by comparing its tip's distance from the wrist to its MCP joint's distance from the wrist. Combined with the thumb-index pinch distance, this drives a small gesture state machine (draw / erase / undo / clear / idle), with undo and clear firing once per gesture *entry* rather than every frame.
- **Stroke smoothing**: raw fingertip samples are jittery at 30fps, so each stroke is rendered with a midpoint-quadratic curve (the standard freehand-drawing technique) instead of straight line segments.
- **Strokes are stored as data** (`{ color, erase, points[] }`), not just painted pixels — that's what makes undo possible: popping the last stroke and replaying the rest from scratch.
- **Live telemetry**: rolling-average FPS and hand-detection confidence are shown in the corner — small, but signals the tracking loop is actually being measured, not just assumed to work.
- **Color picker** and **Save as PNG** round out the base gesture-drawing logic.

## Next steps

- Two-hand support: left hand for a radial color/thickness menu, right hand for drawing.
- "Sketch to Image": send the canvas to an image-generation API and turn the doodle into a polished image.
- Shape recognition: snap rough circles/arrows/boxes to clean vector shapes.
- Persist drawings to local storage or a backend between sessions.
