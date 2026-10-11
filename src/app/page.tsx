"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const COLORS = ["#a855f7", "#ef4444", "#22c55e", "#3b82f6", "#eab308", "#ec4899", "#ffffff", "#171717"];
const BRUSH_SIZES = [3, 7, 12, 18];
const PINCH_ENTER_ABS = 0.05;
const PINCH_EXIT_ABS = 0.07;
// Depth-normalized pinch (thumb-index dist / wrist–middleMCP dist) — stable near/far.
const PINCH_ENTER_RATIO = 0.32;
const PINCH_EXIT_RATIO = 0.42;
const ERASE_RADIUS = 28;
const SMOOTH_WINDOW = 4;
const MIN_MOVE_PX = 3;
// Hold-to-fire: stable frames required, then single fire + cooldown.
const UNDO_HOLD_FRAMES = 10;
const CLEAR_HOLD_FRAMES = 15;
const GESTURE_COOLDOWN_MS = 1000;
const UI_THROTTLE_MS = 250;

type Point = { x: number; y: number };
type Stroke = { color: string; erase: boolean; points: Point[]; width: number };
type Gesture = "draw" | "erase" | "undo" | "clear" | "idle";

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

function isFingerExtended(landmarks: { x: number; y: number }[], tipIdx: number, mcpIdx: number) {
  const wrist = landmarks[WRIST];
  const tip = landmarks[tipIdx];
  const mcp = landmarks[mcpIdx];
  const distTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
  const distMcp = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y);
  return distTip > distMcp * 1.15;
}

function paintStroke(ctx: CanvasRenderingContext2D, stroke: Stroke) {
  const { points, color, erase, width } = stroke;
  if (points.length === 0) return;
  ctx.save();
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  ctx.globalAlpha = 1.0;
  ctx.strokeStyle = color;
  ctx.lineWidth = erase ? ERASE_RADIUS : width;
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

function redrawAll(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, strokes: Stroke[]) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const stroke of strokes) paintStroke(ctx, stroke);
}

// Incremental live segment — O(1) per frame instead of repainting the whole stroke.
function paintSegment(ctx: CanvasRenderingContext2D, stroke: Stroke, fromIdx: number) {
  const { points, color, erase, width } = stroke;
  if (fromIdx < 0 || fromIdx + 1 >= points.length) return;
  ctx.save();
  ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = erase ? ERASE_RADIUS : width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (fromIdx === 0 && points.length === 1) {
    ctx.beginPath();
    ctx.arc(points[0].x, points[0].y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(points[fromIdx].x, points[fromIdx].y);
    ctx.lineTo(points[fromIdx + 1].x, points[fromIdx + 1].y);
    ctx.stroke();
  }
  ctx.restore();
}

export default function AirCanvas() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement>(null);
  const drawCanvasRef = useRef<HTMLCanvasElement>(null);
  const cursorCanvasRef = useRef<HTMLCanvasElement>(null);

  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeColor, setActiveColor] = useState(COLORS[0]);
  const [brushWidth, setBrushWidth] = useState(BRUSH_SIZES[1]);
  const [gestureLabel, setGestureLabel] = useState<Gesture>("idle");
  const [fps, setFps] = useState(0);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [hasHand, setHasHand] = useState(false);
  const [primerDismissed, setPrimerDismissed] = useState(false);
  const [saveWithBg, setSaveWithBg] = useState(false);
  const [holdProgress, setHoldProgress] = useState(0);
  const [touchActive, setTouchActive] = useState(false);

  const activeColorRef = useRef(activeColor);
  const brushWidthRef = useRef(brushWidth);
  const strokesRef = useRef<Stroke[]>([]);
  const activeStrokeRef = useRef<Stroke | null>(null);
  const prevGestureRef = useRef<Gesture>("idle");
  const frameTimesRef = useRef<number[]>([]);
  const undoHoldRef = useRef(0);
  const clearHoldRef = useRef(0);
  const lastUndoFireRef = useRef(0);
  const lastClearFireRef = useRef(0);
  const tipHistoryRef = useRef<Point[]>([]);
  const isPinchingRef = useRef(false);
  const gestureStableRef = useRef<{ gesture: Gesture; count: number }>({ gesture: "idle", count: 0 });
  const lastUiUpdateRef = useRef(0);
  const pendingUiRef = useRef<{ gesture: Gesture; fps: number; conf: number | null; hand: boolean }>({
    gesture: "idle",
    fps: 0,
    conf: null,
    hand: false,
  });
  const touchStrokeRef = useRef<Stroke | null>(null);
  const touchPointerIdRef = useRef<number | null>(null);
  const wakeLockRef = useRef<{ release?: () => Promise<void> } | null>(null);

  useEffect(() => {
    activeColorRef.current = activeColor;
  }, [activeColor]);
  useEffect(() => {
    brushWidthRef.current = brushWidth;
  }, [brushWidth]);

  useEffect(() => {
    if (!primerDismissed) return;
    let handLandmarker: HandLandmarker;
    let animationFrameId: number;
    let stream: MediaStream | null = null;
    let cancelled = false;

    // Keep the screen awake while drawing (best-effort, silent fail).
    const requestWakeLock = async () => {
      try {
        const nav = navigator as Navigator & {
          wakeLock?: { request: (t: string) => Promise<{ release?: () => Promise<void> }> };
        };
        if (nav.wakeLock?.request) {
          wakeLockRef.current = await nav.wakeLock.request("screen");
        }
      } catch {
        /* unsupported or denied — ignore */
      }
    };
    requestWakeLock();
    const onVisibility = () => {
      if (document.visibilityState === "visible") requestWakeLock();
    };
    document.addEventListener("visibilitychange", onVisibility);

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
        setError("Failed to load hand-tracking model. Add hand_landmarker.task to /public/models/.");
      }
    };

    const startCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("This browser does not support camera access.");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(() => {});
            predictWebcam();
          };
        }
      } catch (err) {
        console.error(err);
        setError("Camera permission denied or no camera found.");
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

      if (wasResized) {
        vctx = videoCanvas.getContext("2d")!;
        dctx = drawCanvas.getContext("2d")!;
        if (dctx) {
          redrawAll(drawCanvas, dctx, strokesRef.current);
          if (activeStrokeRef.current) paintStroke(dctx, activeStrokeRef.current);
        }
        if (!vctx || !dctx) return;
      }

      const now = performance.now();
      frameTimesRef.current.push(now);
      if (frameTimesRef.current.length > 20) frameTimesRef.current.shift();
      let currentFps = pendingUiRef.current.fps;
      if (frameTimesRef.current.length >= 2) {
        const span = frameTimesRef.current[frameTimesRef.current.length - 1] - frameTimesRef.current[0];
        currentFps = span > 0 ? Math.round(((frameTimesRef.current.length - 1) * 1000) / span) : 0;
      }

      vctx.save();
      vctx.scale(-1, 1);
      vctx.translate(-videoCanvas.width, 0);
      vctx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
      // subtle dark vignette for better contrast
      vctx.fillStyle = "rgba(0,0,0,0.12)";
      vctx.fillRect(0, 0, videoCanvas.width, videoCanvas.height);
      vctx.restore();

      cctx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);

      const flushUi = (gesture: Gesture, conf: number | null, hand: boolean, progress: number) => {
        pendingUiRef.current = { gesture, fps: currentFps, conf, hand };
        if (now - lastUiUpdateRef.current >= UI_THROTTLE_MS) {
          lastUiUpdateRef.current = now;
          setGestureLabel(pendingUiRef.current.gesture);
          setFps(pendingUiRef.current.fps);
          setConfidence(pendingUiRef.current.conf);
          setHasHand(pendingUiRef.current.hand);
          setHoldProgress(progress);
        }
      };

      const results = handLandmarker.detectForVideo(video, now);

      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        const conf = results.handednesses?.[0]?.[0]?.score ?? null;

        const indexTip = landmarks[INDEX_TIP];
        const thumbTip = landmarks[THUMB_TIP];
        const rawX = (1 - indexTip.x) * drawCanvas.width;
        const rawY = indexTip.y * drawCanvas.height;

        tipHistoryRef.current.push({ x: rawX, y: rawY });
        if (tipHistoryRef.current.length > SMOOTH_WINDOW) tipHistoryRef.current.shift();
        const x = tipHistoryRef.current.reduce((s, p) => s + p.x, 0) / tipHistoryRef.current.length;
        const y = tipHistoryRef.current.reduce((s, p) => s + p.y, 0) / tipHistoryRef.current.length;

        // Depth-normalized pinch: absolute guard + hand-size ratio so near/far both work.
        const pinchAbs = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
        const wrist = landmarks[WRIST];
        const middleMcp = landmarks[MIDDLE_MCP];
        const handSize = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y) || 1;
        const pinchRatio = pinchAbs / handSize;
        if (isPinchingRef.current) {
          if (pinchRatio > PINCH_EXIT_RATIO && pinchAbs > PINCH_EXIT_ABS) isPinchingRef.current = false;
        } else {
          if (pinchRatio < PINCH_ENTER_RATIO && pinchAbs < PINCH_EXIT_ABS) isPinchingRef.current = true;
        }
        const pinching = isPinchingRef.current;

        const indexExt = isFingerExtended(landmarks, INDEX_TIP, INDEX_MCP);
        const middleExt = isFingerExtended(landmarks, MIDDLE_TIP, MIDDLE_MCP);
        const ringExt = isFingerExtended(landmarks, RING_TIP, RING_MCP);
        const pinkyExt = isFingerExtended(landmarks, PINKY_TIP, PINKY_MCP);

        let rawGesture: Gesture = "idle";
        if (pinching) rawGesture = "draw";
        else if (indexExt && middleExt && !ringExt && !pinkyExt) rawGesture = "undo";
        else if (indexExt && middleExt && ringExt && pinkyExt) rawGesture = "erase";
        else if (!indexExt && !middleExt && !ringExt && !pinkyExt) rawGesture = "clear";

        if (rawGesture === gestureStableRef.current.gesture) gestureStableRef.current.count += 1;
        else gestureStableRef.current = { gesture: rawGesture, count: 1 };

        const requiredStable = rawGesture === "draw" || rawGesture === "idle" ? 1 : 3;
        const gesture = gestureStableRef.current.count >= requiredStable ? rawGesture : prevGestureRef.current;

        let progress = 0;
        if (gesture === "draw" || gesture === "erase") {
          undoHoldRef.current = 0;
          clearHoldRef.current = 0;
          if (prevGestureRef.current !== gesture) {
            finishActiveStroke();
            activeStrokeRef.current = {
              color: activeColorRef.current,
              erase: gesture === "erase",
              points: [{ x, y }],
              width: brushWidthRef.current,
            };
            if (activeStrokeRef.current) paintSegment(dctx, activeStrokeRef.current, 0);
          } else if (activeStrokeRef.current) {
            const last = activeStrokeRef.current.points[activeStrokeRef.current.points.length - 1];
            const dx = x - last.x;
            const dy = y - last.y;
            if (dx * dx + dy * dy >= MIN_MOVE_PX * MIN_MOVE_PX) {
              activeStrokeRef.current.points.push({ x, y });
              paintSegment(dctx, activeStrokeRef.current, activeStrokeRef.current.points.length - 2);
            }
          }
        } else {
          finishActiveStroke();
          if (gesture === "undo") {
            undoHoldRef.current += 1;
            progress = Math.min(1, undoHoldRef.current / UNDO_HOLD_FRAMES);
            if (
              undoHoldRef.current >= UNDO_HOLD_FRAMES &&
              prevGestureRef.current !== "undo" &&
              now - lastUndoFireRef.current > GESTURE_COOLDOWN_MS
            ) {
              lastUndoFireRef.current = now;
              strokesRef.current.pop();
              redrawAll(drawCanvas, dctx, strokesRef.current);
            }
          } else undoHoldRef.current = 0;
          if (gesture === "clear") {
            clearHoldRef.current += 1;
            progress = Math.max(progress, Math.min(1, clearHoldRef.current / CLEAR_HOLD_FRAMES));
            if (
              clearHoldRef.current >= CLEAR_HOLD_FRAMES &&
              prevGestureRef.current !== "clear" &&
              now - lastClearFireRef.current > GESTURE_COOLDOWN_MS
            ) {
              lastClearFireRef.current = now;
              strokesRef.current = [];
              dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }
          } else clearHoldRef.current = 0;
        }

        prevGestureRef.current = gesture;
        flushUi(gesture, conf, true, progress);

        cctx.beginPath();
        const cursorRadius = gesture === "erase" ? ERASE_RADIUS / 2 : brushWidthRef.current + 4;
        cctx.arc(x, y, cursorRadius, 0, 2 * Math.PI);
        const cursorColor =
          gesture === "draw"
            ? activeColorRef.current
            : gesture === "erase"
              ? "#f4f4f5"
              : gesture === "undo"
                ? "#3b82f6"
                : gesture === "clear"
                  ? "#ef4444"
                  : "#ffffff";
        cctx.strokeStyle = cursorColor;
        cctx.lineWidth = 2.5;
        cctx.shadowColor = cursorColor;
        cctx.shadowBlur = gesture === "draw" ? 12 : 0;
        if (gesture === "draw") {
          cctx.fillStyle = cursorColor;
          cctx.fill();
        }
        cctx.stroke();
        cctx.shadowBlur = 0;
      } else {
        finishActiveStroke();
        prevGestureRef.current = "idle";
        undoHoldRef.current = 0;
        clearHoldRef.current = 0;
        tipHistoryRef.current = [];
        isPinchingRef.current = false;
        gestureStableRef.current = { gesture: "idle", count: 0 };
        flushUi("idle", null, false, 0);
      }

      animationFrameId = requestAnimationFrame(predictWebcam);
    };

    initializeMediaPipe();
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibility);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (handLandmarker) handLandmarker.close();
      stream?.getTracks().forEach((t) => t.stop());
      wakeLockRef.current?.release?.().catch(() => {});
    };
  }, [primerDismissed]);

  const clearCanvas = () => {
    const canvas = drawCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      strokesRef.current = [];
      touchStrokeRef.current = null;
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
    const videoCanvas = videoCanvasRef.current;
    if (!canvas) return;
    const out = document.createElement("canvas");
    out.width = canvas.width;
    out.height = canvas.height;
    const octx = out.getContext("2d")!;
    if (saveWithBg && videoCanvas && videoCanvas.width > 0) {
      octx.drawImage(videoCanvas, 0, 0, out.width, out.height);
    } else {
      octx.fillStyle = "#0a0a0a";
      octx.fillRect(0, 0, out.width, out.height);
    }
    octx.drawImage(canvas, 0, 0);
    const link = document.createElement("a");
    link.download = `air-canvas-${Date.now()}.png`;
    link.href = out.toDataURL("image/png");
    link.click();
  };

  // Touch / mouse fallback — draw directly when camera/hand is unavailable.
  const canvasPointFromEvent = (e: React.PointerEvent, canvas: HTMLCanvasElement): Point | null => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };
  const onTouchStart = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = drawCanvasRef.current;
    if (!canvas || touchPointerIdRef.current !== null) return;
    const ctx = canvas.getContext("2d");
    const pt = canvasPointFromEvent(e, canvas);
    if (!ctx || !pt) return;
    touchPointerIdRef.current = e.pointerId;
    (e.target as HTMLCanvasElement).setPointerCapture?.(e.pointerId);
    touchStrokeRef.current = {
      color: activeColorRef.current,
      erase: false,
      points: [pt],
      width: Math.max(brushWidthRef.current * 1.5, 6),
    };
    setTouchActive(true);
    paintSegment(ctx, touchStrokeRef.current, 0);
  };
  const onTouchMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = drawCanvasRef.current;
    if (!canvas || e.pointerId !== touchPointerIdRef.current || !touchStrokeRef.current) return;
    const ctx = canvas.getContext("2d");
    const pt = canvasPointFromEvent(e, canvas);
    if (!ctx || !pt) return;
    const pts = touchStrokeRef.current.points;
    const last = pts[pts.length - 1];
    const dx = pt.x - last.x;
    const dy = pt.y - last.y;
    if (dx * dx + dy * dy >= MIN_MOVE_PX * MIN_MOVE_PX) {
      pts.push(pt);
      paintSegment(ctx, touchStrokeRef.current, pts.length - 2);
    }
  };
  const endTouchStroke = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerId !== touchPointerIdRef.current) return;
    touchPointerIdRef.current = null;
    setTouchActive(false);
    if (touchStrokeRef.current && touchStrokeRef.current.points.length > 0) {
      strokesRef.current.push(touchStrokeRef.current);
    }
    touchStrokeRef.current = null;
  };
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  };

  const gestureMeta: Record<Gesture, { label: string; sub: string; color: string; icon: string }> = {
    draw: { label: "Drawing", sub: "Pinch held", color: "text-violet-300", icon: "✦" },
    erase: { label: "Erasing", sub: "Open palm", color: "text-zinc-200", icon: "◯" },
    undo: { label: "Undo", sub: "Hold ✌️", color: "text-blue-300", icon: "↩" },
    clear: { label: "Clear", sub: "Hold ✊", color: "text-red-300", icon: "⌫" },
    idle: {
      label: touchActive ? "Touch draw" : hasHand ? "Idle" : "No hand",
      sub: touchActive ? "Finger on screen" : hasHand ? "Release pinch" : "Show hand or draw with finger",
      color: "text-zinc-400",
      icon: "○",
    },
  };
  const meta = gestureMeta[gestureLabel];

  return (
    <main className="relative flex min-h-dvh w-full flex-col overflow-x-clip bg-[#050508] text-white selection:bg-violet-500/30">
      {/* Background gradients */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-violet-950/30 via-transparent to-blue-950/20" />
        <div className="absolute -top-40 -left-40 h-[600px] w-[600px] rounded-full bg-violet-600/20 blur-[120px]" />
        <div className="absolute -bottom-40 -right-40 h-[700px] w-[700px] rounded-full bg-blue-600/15 blur-[130px]" />
      </div>

      {/* Header */}
      <header className="relative z-20 flex items-center justify-between gap-3 border-b border-white/[0.06] bg-black/20 px-4 py-3 backdrop-blur-xl md:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 shadow-lg shadow-violet-600/20">
            <span className="text-[18px]">🎨</span>
          </div>
          <div>
            <h1 className="text-[15px] font-bold tracking-tight md:text-[17px]">Air Canvas</h1>
            <p className="hidden text-[11px] font-medium tracking-wide text-white/50 md:block">DRAW WITH YOUR HAND · MEDIAPIPE</p>
          </div>
          <span className="ml-2 hidden items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.06] px-2.5 py-1 text-[11px] font-semibold tracking-wide text-white/70 backdrop-blur md:flex">
            <span className={`h-2 w-2 rounded-full ${isLoaded ? "bg-emerald-400 shadow shadow-emerald-400/50" : "animate-pulse bg-amber-400"}`} />
            {isLoaded ? "LIVE" : "LOADING"}
          </span>
        </div>

        <div className="hidden items-center gap-2 lg:flex">
          <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-1 py-1 backdrop-blur">
            <span className="px-2 text-[11px] font-semibold text-white/60">FPS</span>
            <span className="rounded-full bg-white px-2 py-0.5 text-[12px] font-bold text-black">{fps}</span>
            {confidence !== null && (
              <>
                <span className="ml-1 h-3 w-px bg-white/10" />
                <span className="px-2 text-[11px] font-semibold text-white/60">{Math.round(confidence * 100)}% conf</span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleFullscreen}
            className="hidden rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold backdrop-blur transition hover:bg-white/10 md:inline-flex"
          >
            ⛶ Fullscreen
          </button>
          <a
            href="https://github.com"
            target="_blank"
            className="hidden rounded-xl bg-white px-4 py-2 text-xs font-bold text-black transition hover:bg-zinc-100 md:inline-flex"
          >
            Share
          </a>
        </div>
      </header>

      {/* Mobile status bar — hidden in landscape to save vertical space */}
      <div className="relative z-10 flex items-center justify-between border-b border-white/[0.04] bg-black/10 px-4 py-2 backdrop-blur portrait:flex landscape:hidden md:hidden">
        <span className="truncate text-xs text-white/60">{error ? error : !isLoaded ? "Loading AI models…" : touchActive ? "Touch drawing…" : "Pinch to draw · or touch screen"}</span>
        <span className="ml-2 shrink-0 text-xs font-bold">{fps} fps</span>
      </div>

      <video ref={videoRef} className="hidden" playsInline muted autoPlay disablePictureInPicture />

      {/* Canvas area */}
      <div className="relative z-10 mx-auto flex w-full max-w-[1280px] flex-1 flex-col gap-3 p-3 md:gap-4 md:p-6">
        {/* Fits both orientations: tall in portrait, height-capped 16:9 in landscape. No rotation needed. */}
        <div className="relative mx-auto w-full max-w-full overflow-hidden rounded-[20px] border border-white/10 bg-black shadow-[0_20px_80px_rgba(0,0,0,0.6)] portrait:aspect-[3/4] portrait:max-h-[calc(100dvh-170px)] landscape:mx-auto landscape:h-[calc(100dvh-150px)] landscape:w-auto landscape:aspect-[16/9] landscape:max-w-full sm:portrait:aspect-[4/3] md:h-auto md:max-h-none md:portrait:aspect-[16/10] md:portrait:max-h-none md:landscape:h-auto md:landscape:w-full md:landscape:aspect-[16/10] md:rounded-[28px] lg:aspect-[16/8.5]">
          <canvas ref={videoCanvasRef} className="absolute inset-0 block h-full w-full" />
          <canvas
            ref={drawCanvasRef}
            className="absolute inset-0 block h-full w-full cursor-crosshair"
            style={{ touchAction: "none" }}
            onPointerDown={onTouchStart}
            onPointerMove={onTouchMove}
            onPointerUp={endTouchStroke}
            onPointerCancel={endTouchStroke}
          />
          <canvas ref={cursorCanvasRef} className="pointer-events-none absolute inset-0 block h-full w-full" />

          {/* Top left — color + brush: compact horizontal strip on phones, vertical panel on desktop */}
          <div className="canvas-overlay absolute left-2 top-2 md:left-4 md:top-4">
            <div className="flex max-w-[46vw] flex-row items-center gap-2 overflow-x-auto rounded-2xl border border-white/10 bg-black/40 px-2 py-2 backdrop-blur-xl landscape:max-w-[36vw] sm:max-w-none md:max-w-none md:flex-col md:gap-2 md:px-2.5 md:py-3">
              <span className="hidden text-[10px] font-bold tracking-widest text-white/40 md:block">COLOR</span>
              <div className="grid shrink-0 grid-cols-4 gap-1.5 md:grid-cols-2 md:gap-2">
                {COLORS.map((c) => (
                  <button
                    key={c}
                    onClick={() => setActiveColor(c)}
                    className={`h-6 w-6 rounded-full border-2 transition-all md:h-8 md:w-8 ${activeColor === c ? "scale-110 border-white shadow-lg" : "border-white/10 hover:border-white/30"}`}
                    style={{ backgroundColor: c, boxShadow: activeColor === c ? `0 0 12px ${c}80` : undefined }}
                    aria-label={`Color ${c}`}
                  />
                ))}
              </div>
              <div className="h-8 w-px shrink-0 bg-white/10 md:h-px md:w-full" />
              <span className="hidden text-[10px] font-bold tracking-widest text-white/40 md:block">SIZE</span>
              <div className="flex shrink-0 flex-row gap-1.5 md:flex-col">
                {BRUSH_SIZES.map((s) => (
                  <button
                    key={s}
                    onClick={() => setBrushWidth(s)}
                    className={`flex h-6 w-6 items-center justify-center rounded-full border transition md:h-8 md:w-8 ${brushWidth === s ? "border-white bg-white text-black" : "border-white/10 bg-white/5 text-white/70 hover:bg-white/10"}`}
                  >
                    <span className="rounded-full bg-current" style={{ width: Math.min(s, 12), height: Math.min(s, 12) }} />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Top right — gesture status: compact on phones, with hold progress */}
          <div className="absolute right-2 top-2 flex flex-col items-end gap-2 md:right-4 md:top-4">
            <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-black/40 px-2.5 py-2 backdrop-blur-xl md:gap-3 md:px-4 md:py-3">
              <div className={`flex h-8 w-8 items-center justify-center rounded-xl border bg-white/5 text-base md:h-10 md:w-10 md:text-lg ${meta.color} border-white/10`}>{meta.icon}</div>
              <div className="text-right leading-tight">
                <div className={`text-xs font-bold md:text-sm ${meta.color}`}>{meta.label}</div>
                <div className="hidden text-xs text-white/50 sm:block">{meta.sub}</div>
                {(gestureLabel === "undo" || gestureLabel === "clear") && holdProgress > 0 && (
                  <div className="mt-1 h-1 w-20 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full rounded-full bg-current transition-[width]" style={{ width: `${Math.round(holdProgress * 100)}%` }} />
                  </div>
                )}
              </div>
              <div className={`ml-1 h-2 w-2 rounded-full ${gestureLabel === "draw" ? "animate-pulse bg-violet-400" : hasHand ? "bg-emerald-400" : "bg-zinc-600"}`} />
            </div>
            <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-black/40 px-3 py-1.5 text-xs backdrop-blur-xl md:flex">
              <span className="text-white/50">FPS</span>
              <span className="font-bold">{fps}</span>
              <span className="h-3 w-px bg-white/10" />
              <span className="text-white/50">Conf</span>
              <span className="font-bold">{confidence !== null ? `${Math.round(confidence * 100)}%` : "—"}</span>
            </div>
          </div>

          {/* Bottom center — legend: portrait only on phones (saves height in landscape), always on md+ */}
          <div className="absolute bottom-[68px] left-1/2 flex -translate-x-1/2 items-center gap-1 whitespace-nowrap rounded-full border border-white/10 bg-black/50 px-2 py-1.5 text-[11px] font-medium backdrop-blur-xl portrait:flex landscape:hidden sm:bottom-[72px] md:bottom-4 md:gap-2 md:px-4 md:text-xs md:portrait:flex md:landscape:flex">
            <span className="flex items-center gap-1.5 whitespace-nowrap">
              <span className="rounded-full bg-violet-600 px-2 py-0.5 text-[10px] font-bold text-white">PINCH</span>
              <span className="hidden text-white/80 md:inline">Draw</span>
            </span>
            <span className="h-4 w-px bg-white/10" />
            <span className="whitespace-nowrap text-white/70">🖐 Erase</span>
            <span className="h-4 w-px bg-white/10" />
            <span className="whitespace-nowrap text-white/70">✌️ Undo</span>
            <span className="h-4 w-px bg-white/10" />
            <span className="whitespace-nowrap text-white/70">✊ Clear</span>
          </div>

          {/* Bottom actions: full-width row in portrait, compact floating-right in landscape */}
          <div className="canvas-overlay absolute inset-x-2 bottom-2 flex items-center justify-end gap-1.5 portrait:inset-x-2 landscape:inset-x-auto landscape:right-2 landscape:bottom-2 md:inset-x-auto md:bottom-4 md:right-4 md:gap-2">
            <button
              onClick={() => setSaveWithBg((v) => !v)}
              title={saveWithBg ? "Save includes camera background" : "Save strokes only"}
              className={`rounded-full border px-3 py-2 text-xs font-bold backdrop-blur-xl transition landscape:flex-none sm:flex-none md:px-4 md:py-2.5 ${saveWithBg ? "border-violet-400/40 bg-violet-600/80" : "border-white/10 bg-black/50 hover:bg-white/10"}`}
            >
              {saveWithBg ? "🖼 BG on" : "BG off"}
            </button>
            <button onClick={undoLast} className="flex-1 rounded-full border border-white/10 bg-black/50 px-3 py-2 text-xs font-bold backdrop-blur-xl transition hover:bg-white/10 landscape:flex-none sm:flex-none md:px-5 md:py-2.5">
              ↩ Undo
            </button>
            <button onClick={clearCanvas} className="flex-1 rounded-full border border-red-500/20 bg-red-600/90 px-3 py-2 text-xs font-bold backdrop-blur-xl transition hover:bg-red-600 landscape:flex-none sm:flex-none md:px-5 md:py-2.5">
              Clear
            </button>
            <button onClick={saveDrawing} className="flex-1 rounded-full bg-white px-4 py-2 text-xs font-bold text-black shadow-lg transition hover:bg-zinc-100 landscape:flex-none sm:flex-none md:px-5 md:py-2.5">
              ⬇ Save
            </button>
          </div>

          {/* Primer / loading / error overlay */}
          {(!primerDismissed || !isLoaded || error) && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6">
              <div className="w-full max-w-md rounded-3xl border border-white/10 bg-zinc-900 p-6 text-center shadow-2xl">
                {!primerDismissed ? (
                  <>
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-600 to-indigo-600 text-2xl">
                      🎨
                    </div>
                    <h3 className="text-lg font-bold">Draw in the air</h3>
                    <p className="mt-2 text-sm leading-relaxed text-white/60">
                      Pinch thumb + index to draw · open palm 🖐 to erase · ✌️ holds undo · ✊ holds clear. No camera? Just draw with your finger on the canvas.
                    </p>
                    <p className="mt-3 text-xs text-white/30">Needs HTTPS + camera permission (iOS Safari: allow camera when asked).</p>
                    <button
                      onClick={() => setPrimerDismissed(true)}
                      className="mt-4 rounded-full bg-white px-6 py-2.5 text-sm font-bold text-black transition hover:bg-zinc-100"
                    >
                      Enable camera + start
                    </button>
                    <button
                      onClick={() => setPrimerDismissed(true)}
                      className="mt-2 block w-full text-xs font-medium text-white/40 underline-offset-4 hover:underline"
                    >
                      Skip camera — touch draw only
                    </button>
                  </>
                ) : !error ? (
                  <>
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-600 to-indigo-600">
                      <span className="h-6 w-6 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    </div>
                    <h3 className="text-lg font-bold">Loading hand tracking…</h3>
                    <p className="mt-2 text-sm leading-relaxed text-white/60">
                      Downloading MediaPipe model (~8 MB). Please allow camera access when prompted.
                    </p>
                    <p className="mt-3 text-xs text-white/30">First load may take 5–10 seconds</p>
                  </>
                ) : (
                  <>
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-red-500/20 text-xl">⚠️</div>
                    <h3 className="text-lg font-bold text-red-300">Something went wrong</h3>
                    <p className="mt-2 text-sm leading-relaxed text-white/60">{error}</p>
                    <p className="mt-2 text-xs text-white/40">You can still draw with touch even without the camera.</p>
                    <div className="mt-4 flex items-center justify-center gap-2">
                      <button onClick={() => setError(null)} className="rounded-full border border-white/10 bg-white/5 px-5 py-2 text-sm font-bold">
                        Continue offline
                      </button>
                      <button onClick={() => window.location.reload()} className="rounded-full bg-white px-5 py-2 text-sm font-bold text-black">
                        Retry
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Helper bar below canvas — hidden on short landscape phones to let canvas fit, shown otherwise */}
        <div className="flex-col items-center justify-between gap-3 rounded-2xl border border-white/5 bg-white/[0.03] px-4 py-3 backdrop-blur portrait:flex md:flex-row landscape:hidden md:flex">
          <p className="text-center text-xs leading-relaxed text-white/50 md:text-left">
            <span className="font-semibold text-white/80">Tip:</span> Pinch to draw · open palm to erase · or draw with finger on touch · keep hand centered and well-lit
          </p>
          <div className="flex items-center gap-2 text-xs">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-medium text-white/60">Next.js 16 · MediaPipe · Tailwind 4</span>
          </div>
        </div>

        {/* Signature */}
        <footer className="flex items-center justify-center gap-2 py-4 text-center text-xs text-white/40">
          <span>Crafted with passion by</span>
          <a
            href="https://www.linkedin.com/in/takbir-zaman-bhuiyan/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 font-semibold text-white/70 underline decoration-white/20 underline-offset-4 transition hover:text-violet-300 hover:decoration-violet-300/50"
          >
            Takbir Zaman Bhuiyan
            <svg aria-hidden="true" viewBox="0 0 24 24" className="h-3.5 w-3.5 fill-current opacity-70">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.777 13.019H3.56V9h3.554v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
            </svg>
          </a>
          <span className="hidden sm:inline text-white/20">·</span>
          <span className="hidden sm:inline">© {new Date().getFullYear()} Air Canvas</span>
        </footer>
      </div>

      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'); *{font-family: Inter, system-ui, sans-serif}`}</style>
    </main>
  );
}
