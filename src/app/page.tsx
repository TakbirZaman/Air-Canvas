"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const COLORS = ["#a855f7", "#ef4444", "#22c55e", "#3b82f6", "#eab308", "#ffffff"];
const PINCH_ENTER = 0.05; // distance to start drawing - tight pinch required
const PINCH_EXIT = 0.07; // distance to stop drawing - hysteresis prevents flicker
const DRAW_WIDTH = 9;
const ERASE_RADIUS = 28;
const SMOOTH_WINDOW = 4; // average last N fingertip positions
const MIN_MOVE_PX = 3; // ignore tiny jitter < 3px

type Point = { x: number; y: number };
type Stroke = { color: string; erase: boolean; points: Point[] };
type Gesture = "draw" | "erase" | "undo" | "clear" | "idle";

// Landmark indices (MediaPipe Hand Landmarker, 21 points per hand)
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_TIP = 12;
const RING_MCP = 13;
const RING_TIP = 16;
const PINKY_MCP = 17;
const PINKY_TIP = 20;

function isFingerExtended(
  landmarks: { x: number; y: number }[],
  tipIdx: number,
  mcpIdx: number
) {
  const wrist = landmarks[WRIST];
  const tip = landmarks[tipIdx];
  const mcp = landmarks[mcpIdx];
  const distTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
  const distMcp = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y);
  return distTip > distMcp * 1.15;
}

// Draws a stroke's full point history using a midpoint-quadratic smoothing
// technique — turns jittery raw fingertip samples into a smooth curve.
function paintStroke(ctx: CanvasRenderingContext2D, stroke: Stroke) {
  const { points, color, erase } = stroke;
  if (points.length === 0) return;

  ctx.save();
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  ctx.globalAlpha = 1.0;
  ctx.strokeStyle = color;
  ctx.lineWidth = erase ? ERASE_RADIUS : DRAW_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (points.length < 3) {
    ctx.beginPath();
    ctx.arc(points[0].x, points[0].y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    if (!erase) ctx.fill();
    if (points.length === 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[1].x, points[1].y);
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length - 1; i++) {
    const p1 = points[i];
    const p2 = points[i + 1];
    const midX = (p1.x + p2.x) / 2;
    const midY = (p1.y + p2.y) / 2;
    ctx.quadraticCurveTo(p1.x, p1.y, midX, midY);
  }
  ctx.stroke();
  ctx.restore();
}

function redrawAll(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  strokes: Stroke[]
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const stroke of strokes) paintStroke(ctx, stroke);
}

export default function AirCanvas() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement>(null); // layer 1: webcam feed
  const drawCanvasRef = useRef<HTMLCanvasElement>(null); // layer 2: persisted strokes
  const cursorCanvasRef = useRef<HTMLCanvasElement>(null); // layer 3: fingertip indicator (cleared every frame)

  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeColor, setActiveColor] = useState(COLORS[0]);
  const [gestureLabel, setGestureLabel] = useState("—");
  const [fps, setFps] = useState(0);
  const [confidence, setConfidence] = useState<number | null>(null);

  const activeColorRef = useRef(activeColor);
  const strokesRef = useRef<Stroke[]>([]);
  const activeStrokeRef = useRef<Stroke | null>(null);
  const prevGestureRef = useRef<Gesture>("idle");
  const frameTimesRef = useRef<number[]>([]);
  // Hold counters to prevent accidental undo/clear when briefly showing gesture
  const undoHoldRef = useRef(0);
  const clearHoldRef = useRef(0);
  // Smoothing & hysteresis to reduce sensor sensitivity / jitter
  const tipHistoryRef = useRef<Point[]>([]);
  const isPinchingRef = useRef(false);
  const gestureStableRef = useRef<{ gesture: Gesture; count: number }>({ gesture: "idle", count: 0 });

  useEffect(() => {
    activeColorRef.current = activeColor;
  }, [activeColor]);

  useEffect(() => {
    let handLandmarker: HandLandmarker;
    let animationFrameId: number;
    let stream: MediaStream | null = null;
    let cancelled = false;

    const initializeMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });

        if (cancelled) return;
        setIsLoaded(true);
        await startCamera();
      } catch (err) {
        console.error(err);
        setError(
          "Failed to load the hand-tracking model. Make sure hand_landmarker.task is in /public/models/."
        );
      }
    };

    const startCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("This browser does not support camera access.");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: "user" },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            predictWebcam();
          };
        }
      } catch (err) {
        console.error(err);
        setError("Camera permission was denied or no camera was found.");
      }
    };

    const finishActiveStroke = () => {
      if (activeStrokeRef.current && activeStrokeRef.current.points.length > 0) {
        strokesRef.current.push(activeStrokeRef.current);
      }
      activeStrokeRef.current = null;
    };

    const predictWebcam = () => {
      const video = videoRef.current;
      const videoCanvas = videoCanvasRef.current;
      const drawCanvas = drawCanvasRef.current;
      const cursorCanvas = cursorCanvasRef.current;
      if (!video || !videoCanvas || !drawCanvas || !cursorCanvas || !handLandmarker) return;

      // Handle canvas resize BEFORE getting contexts — changing width/height
      // resets the canvas and invalidates any previously obtained context.
      // We must preserve and redraw strokes so drawings are never lost.
      let wasResized = false;
      if (videoCanvas.width !== video.videoWidth && video.videoWidth > 0) {
        videoCanvas.width = video.videoWidth;
        videoCanvas.height = video.videoHeight;
        drawCanvas.width = video.videoWidth;
        drawCanvas.height = video.videoHeight;
        cursorCanvas.width = video.videoWidth;
        cursorCanvas.height = video.videoHeight;
        wasResized = true;
      }

      let vctx = videoCanvas.getContext("2d");
      let dctx = drawCanvas.getContext("2d");
      const cctx = cursorCanvas.getContext("2d");
      if (!vctx || !dctx || !cctx) return;

      // After resize the draw canvas was cleared — immediately restore all strokes
      // plus the in-progress active stroke so user drawings persist.
      if (wasResized) {
        vctx = videoCanvas.getContext("2d")!;
        dctx = drawCanvas.getContext("2d")!;
        if (dctx) {
          redrawAll(drawCanvas, dctx, strokesRef.current);
          if (activeStrokeRef.current) paintStroke(dctx, activeStrokeRef.current);
        }
        if (!vctx || !dctx) return;
      }

      // --- FPS tracking (rolling average over last ~20 frames) ---
      const now = performance.now();
      frameTimesRef.current.push(now);
      if (frameTimesRef.current.length > 20) frameTimesRef.current.shift();
      if (frameTimesRef.current.length >= 2) {
        const span = frameTimesRef.current[frameTimesRef.current.length - 1] - frameTimesRef.current[0];
        const avgFps = span > 0 ? ((frameTimesRef.current.length - 1) * 1000) / span : 0;
        setFps(Math.round(avgFps));
      }

      // Layer 1: repaint the mirrored webcam frame every tick — safe, nothing to preserve here.
      vctx.save();
      vctx.scale(-1, 1);
      vctx.translate(-videoCanvas.width, 0);
      vctx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
      vctx.restore();

      // Layer 3: fingertip indicator gets cleared every tick — this is what actually
      // fixes the "ghost circles" issue, since it never touches layer 2 (the strokes).
      cctx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);

      const results = handLandmarker.detectForVideo(video, now);

      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        setConfidence(results.handednesses?.[0]?.[0]?.score ?? null);

        const indexTip = landmarks[INDEX_TIP];
        const thumbTip = landmarks[THUMB_TIP];
        const rawX = (1 - indexTip.x) * drawCanvas.width;
        const rawY = indexTip.y * drawCanvas.height;

        // --- Smoothing: moving average over last N positions to reduce jitter ---
        tipHistoryRef.current.push({ x: rawX, y: rawY });
        if (tipHistoryRef.current.length > SMOOTH_WINDOW) tipHistoryRef.current.shift();
        const x = tipHistoryRef.current.reduce((s, p) => s + p.x, 0) / tipHistoryRef.current.length;
        const y = tipHistoryRef.current.reduce((s, p) => s + p.y, 0) / tipHistoryRef.current.length;

        // --- Pinch with hysteresis: prevents flicker when distance near threshold ---
        const pinchDistance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
        if (isPinchingRef.current) {
          if (pinchDistance > PINCH_EXIT) isPinchingRef.current = false;
        } else {
          if (pinchDistance < PINCH_ENTER) isPinchingRef.current = true;
        }
        const pinching = isPinchingRef.current;

        const indexExt = isFingerExtended(landmarks, INDEX_TIP, INDEX_MCP);
        const middleExt = isFingerExtended(landmarks, MIDDLE_TIP, MIDDLE_MCP);
        const ringExt = isFingerExtended(landmarks, RING_TIP, RING_MCP);
        const pinkyExt = isFingerExtended(landmarks, PINKY_TIP, PINKY_MCP);

        // Gesture detection with stability filter - 5-finger erase removed per user request (was disturbing)
        let rawGesture: Gesture = "idle";
        if (pinching) {
          rawGesture = "draw";
        } else if (indexExt && middleExt && !ringExt && !pinkyExt) {
          rawGesture = "undo";
        } else if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
          rawGesture = "clear";
        }
        if (rawGesture === gestureStableRef.current.gesture) {
          gestureStableRef.current.count += 1;
        } else {
          gestureStableRef.current = { gesture: rawGesture, count: 1 };
        }
        // Draw reacts instantly (1 frame), others require 3 frames stability to avoid sensitivity
        const requiredStable = rawGesture === "draw" || rawGesture === "idle" ? 1 : 3;
        const gesture = gestureStableRef.current.count >= requiredStable ? rawGesture : prevGestureRef.current;

        // Continuous gestures (draw / erase): append to the active stroke.
        // FIX: drawings must persist when you release pinch (go to idle / no hand).
        // We only ever clear the draw canvas on explicit clear or via erase strokes.
        if (gesture === "draw" || gesture === "erase") {
          // reset one-shot hold counters when in continuous mode
          undoHoldRef.current = 0;
          clearHoldRef.current = 0;
          if (prevGestureRef.current !== gesture) {
            finishActiveStroke();
            activeStrokeRef.current = {
              color: activeColorRef.current,
              erase: gesture === "erase",
              points: [{ x, y }],
            };
          } else if (activeStrokeRef.current) {
            const last = activeStrokeRef.current.points[activeStrokeRef.current.points.length - 1];
            const dx = x - last.x;
            const dy = y - last.y;
            // Ignore micro-movements < MIN_MOVE_PX to reduce sensor jitter
            if (dx * dx + dy * dy >= MIN_MOVE_PX * MIN_MOVE_PX) {
              activeStrokeRef.current.points.push({ x, y });
            }
          }
          if (activeStrokeRef.current) paintStroke(dctx, activeStrokeRef.current);
        } else {
          finishActiveStroke();
          // Debounced one-shot gestures: require holding pose for ~8 frames (~250ms)
          // so brief transitions after releasing pinch don't accidentally undo/clear.
          if (gesture === "undo") {
            undoHoldRef.current += 1;
            if (undoHoldRef.current === 8 && prevGestureRef.current !== "undo") {
              strokesRef.current.pop();
              redrawAll(drawCanvas, dctx, strokesRef.current);
            }
          } else {
            undoHoldRef.current = 0;
          }
          if (gesture === "clear") {
            clearHoldRef.current += 1;
            if (clearHoldRef.current === 12 && prevGestureRef.current !== "clear") {
              strokesRef.current = [];
              dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }
          } else {
            clearHoldRef.current = 0;
          }
          // idle does nothing — strokes remain untouched
        }

        prevGestureRef.current = gesture;
        setGestureLabel(
          { draw: "Drawing", erase: "Erasing", undo: "Undo ✌️", clear: "Cleared ✊", idle: "Idle" }[gesture]
        );

        // Cursor indicator reflects the current gesture.
        cctx.beginPath();
        const cursorRadius = gesture === "erase" ? ERASE_RADIUS / 2 : 8;
        cctx.arc(x, y, cursorRadius, 0, 2 * Math.PI);
        const cursorColor =
          gesture === "draw"
            ? activeColorRef.current
            : gesture === "erase"
            ? "#ffffff"
            : gesture === "undo"
            ? "#3b82f6"
            : gesture === "clear"
            ? "#ef4444"
            : "#9ca3af";
        cctx.strokeStyle = cursorColor;
        cctx.lineWidth = 2;
        gesture === "draw" ? (cctx.fillStyle = cursorColor) : null;
        if (gesture === "draw") cctx.fill();
        cctx.stroke();
      } else {
        finishActiveStroke();
        prevGestureRef.current = "idle";
        undoHoldRef.current = 0;
        clearHoldRef.current = 0;
        tipHistoryRef.current = [];
        isPinchingRef.current = false;
        gestureStableRef.current = { gesture: "idle", count: 0 };
        setGestureLabel("No hand detected");
        setConfidence(null);
      }

      animationFrameId = requestAnimationFrame(predictWebcam);
    };

    initializeMediaPipe();

    return () => {
      cancelled = true;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (handLandmarker) handLandmarker.close();
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const clearCanvas = () => {
    const canvas = drawCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      strokesRef.current = [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const undoLast = () => {
    const canvas = drawCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      strokesRef.current.pop();
      redrawAll(canvas, ctx, strokesRef.current);
    }
  };

  const saveDrawing = () => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = "air-canvas.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  return (
    <main className="flex h-screen w-screen flex-col bg-gray-900 text-white overflow-hidden">
      {/* Top bar */}
      <div className="flex shrink-0 items-center justify-between px-4 py-2 bg-gray-900">
        <h1 className="text-xl font-bold whitespace-nowrap">Air Canvas 🎨</h1>
        <p className="hidden sm:block text-xs text-gray-400 text-center flex-1 px-4">
          {error
            ? error
            : !isLoaded
            ? "Loading AI Models..."
            : "Pinch 2 fingers to draw · ✌️ undo · ✊ clear"}
        </p>
        <button
          onClick={toggleFullscreen}
          className="shrink-0 bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded-lg text-sm font-semibold transition"
          title="Toggle fullscreen"
        >
          ⛶ Fullscreen
        </button>
      </div>
      <p className="sm:hidden shrink-0 text-xs text-gray-400 text-center px-4 pb-2">
        {error
          ? error
          : !isLoaded
          ? "Loading AI Models..."
          : "Pinch 2 fingers to draw"}
      </p>

      <video ref={videoRef} className="hidden" playsInline />

      <div className="relative flex-1 w-full overflow-hidden bg-black">
        <canvas ref={videoCanvasRef} className="absolute inset-0 w-full h-full object-cover" />
        <canvas ref={drawCanvasRef} className="absolute inset-0 w-full h-full object-cover" />
        <canvas ref={cursorCanvasRef} className="absolute inset-0 w-full h-full object-cover" />

        {/* Color picker */}
        <div className="absolute top-4 left-4 flex gap-2 bg-black/40 backdrop-blur px-3 py-2 rounded-lg">
          {COLORS.map((color) => (
            <button
              key={color}
              onClick={() => setActiveColor(color)}
              className={`w-6 h-6 rounded-full border-2 transition ${
                activeColor === color ? "border-white scale-110" : "border-transparent"
              }`}
              style={{ backgroundColor: color }}
              aria-label={`Select color ${color}`}
            />
          ))}
        </div>

        {/* Live status: gesture, FPS, confidence */}
        <div className="absolute top-4 right-4 bg-black/40 backdrop-blur px-3 py-2 rounded-lg text-xs text-right leading-tight">
          <div className="font-semibold">{gestureLabel}</div>
          <div className="text-gray-300">
            {fps} fps{confidence !== null ? ` · ${Math.round(confidence * 100)}% conf` : ""}
          </div>
        </div>

        <div className="absolute bottom-4 right-4 flex gap-2">
          <button
            onClick={undoLast}
            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-semibold transition"
          >
            Undo
          </button>
          <button
            onClick={saveDrawing}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-semibold transition"
          >
            Save
          </button>
          <button
            onClick={clearCanvas}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-semibold transition"
          >
            Clear
          </button>
        </div>

        {/* Bottom legend */}
        <div className="absolute bottom-14 left-1/2 -translate-x-1/2 flex gap-4 text-xs text-gray-300 bg-black/40 backdrop-blur px-3 py-1.5 rounded-lg whitespace-nowrap">
          <span>2 fingers - draw</span>
          <span>undo</span>
          <span>clear</span>
        </div>
      </div>
    </main>
  );
}
