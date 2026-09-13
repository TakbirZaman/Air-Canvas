"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const COLORS = ["#a855f7", "#ef4444", "#22c55e", "#3b82f6", "#eab308", "#ec4899", "#ffffff", "#171717"];
const BRUSH_SIZES = [3, 7, 12, 18];
const PINCH_ENTER = 0.05;
const PINCH_EXIT = 0.07;
const ERASE_RADIUS = 28;
const SMOOTH_WINDOW = 4;
const MIN_MOVE_PX = 3;

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

  const activeColorRef = useRef(activeColor);
  const brushWidthRef = useRef(brushWidth);
  const strokesRef = useRef<Stroke[]>([]);
  const activeStrokeRef = useRef<Stroke | null>(null);
  const prevGestureRef = useRef<Gesture>("idle");
  const frameTimesRef = useRef<number[]>([]);
  const undoHoldRef = useRef(0);
  const clearHoldRef = useRef(0);
  const tipHistoryRef = useRef<Point[]>([]);
  const isPinchingRef = useRef(false);
  const gestureStableRef = useRef<{ gesture: Gesture; count: number }>({ gesture: "idle", count: 0 });

  useEffect(() => {
    activeColorRef.current = activeColor;
  }, [activeColor]);
  useEffect(() => {
    brushWidthRef.current = brushWidth;
  }, [brushWidth]);

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
      if (frameTimesRef.current.length >= 2) {
        const span = frameTimesRef.current[frameTimesRef.current.length - 1] - frameTimesRef.current[0];
        const avgFps = span > 0 ? ((frameTimesRef.current.length - 1) * 1000) / span : 0;
        setFps(Math.round(avgFps));
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

      const results = handLandmarker.detectForVideo(video, now);

      if (results.landmarks && results.landmarks.length > 0) {
        setHasHand(true);
        const landmarks = results.landmarks[0];
        setConfidence(results.handednesses?.[0]?.[0]?.score ?? null);

        const indexTip = landmarks[INDEX_TIP];
        const thumbTip = landmarks[THUMB_TIP];
        const rawX = (1 - indexTip.x) * drawCanvas.width;
        const rawY = indexTip.y * drawCanvas.height;

        tipHistoryRef.current.push({ x: rawX, y: rawY });
        if (tipHistoryRef.current.length > SMOOTH_WINDOW) tipHistoryRef.current.shift();
        const x = tipHistoryRef.current.reduce((s, p) => s + p.x, 0) / tipHistoryRef.current.length;
        const y = tipHistoryRef.current.reduce((s, p) => s + p.y, 0) / tipHistoryRef.current.length;

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

        let rawGesture: Gesture = "idle";
        if (pinching) rawGesture = "draw";
        else if (indexExt && middleExt && !ringExt && !pinkyExt) rawGesture = "undo";
        else if (!indexExt && !middleExt && !ringExt && !pinkyExt) rawGesture = "clear";

        if (rawGesture === gestureStableRef.current.gesture) gestureStableRef.current.count += 1;
        else gestureStableRef.current = { gesture: rawGesture, count: 1 };

        const requiredStable = rawGesture === "draw" || rawGesture === "idle" ? 1 : 3;
        const gesture = gestureStableRef.current.count >= requiredStable ? rawGesture : prevGestureRef.current;

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
          } else if (activeStrokeRef.current) {
            const last = activeStrokeRef.current.points[activeStrokeRef.current.points.length - 1];
            const dx = x - last.x;
            const dy = y - last.y;
            if (dx * dx + dy * dy >= MIN_MOVE_PX * MIN_MOVE_PX) activeStrokeRef.current.points.push({ x, y });
          }
          if (activeStrokeRef.current) paintStroke(dctx, activeStrokeRef.current);
        } else {
          finishActiveStroke();
          if (gesture === "undo") {
            undoHoldRef.current += 1;
            if (undoHoldRef.current === 8 && prevGestureRef.current !== "undo") {
              strokesRef.current.pop();
              redrawAll(drawCanvas, dctx, strokesRef.current);
            }
          } else undoHoldRef.current = 0;
          if (gesture === "clear") {
            clearHoldRef.current += 1;
            if (clearHoldRef.current === 12 && prevGestureRef.current !== "clear") {
              strokesRef.current = [];
              dctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            }
          } else clearHoldRef.current = 0;
        }

        prevGestureRef.current = gesture;
        setGestureLabel(gesture);

        cctx.beginPath();
        const cursorRadius = gesture === "erase" ? ERASE_RADIUS / 2 : brushWidthRef.current + 4;
        cctx.arc(x, y, cursorRadius, 0, 2 * Math.PI);
        const cursorColor =
          gesture === "draw" ? activeColorRef.current : gesture === "undo" ? "#3b82f6" : gesture === "clear" ? "#ef4444" : "#ffffff";
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
        setGestureLabel("idle");
        setHasHand(false);
        setConfidence(null);
      }

      animationFrameId = requestAnimationFrame(predictWebcam);
    };

    initializeMediaPipe();
    return () => {
      cancelled = true;
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (handLandmarker) handLandmarker.close();
      stream?.getTracks().forEach((t) => t.stop());
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
    // composite: create offscreen canvas with black background + strokes
    const out = document.createElement("canvas");
    out.width = canvas.width;
    out.height = canvas.height;
    const octx = out.getContext("2d")!;
    octx.fillStyle = "#0a0a0a";
    octx.fillRect(0, 0, out.width, out.height);
    octx.drawImage(canvas, 0, 0);
    const link = document.createElement("a");
    link.download = `air-canvas-${Date.now()}.png`;
    link.href = out.toDataURL("image/png");
    link.click();
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
    idle: { label: hasHand ? "Idle" : "No hand", sub: hasHand ? "Release pinch" : "Show your hand", color: "text-zinc-400", icon: "○" },
  };
  const meta = gestureMeta[gestureLabel];

  return (
    <main className="relative min-h-screen w-screen overflow-hidden bg-[#050508] text-white selection:bg-violet-500/30">
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

      {/* Mobile status bar */}
      <div className="relative z-10 flex items-center justify-between border-b border-white/[0.04] bg-black/10 px-4 py-2 backdrop-blur md:hidden">
        <span className="text-xs text-white/60">{error ? error : !isLoaded ? "Loading AI models…" : "Pinch thumb + index to draw"}</span>
        <span className="text-xs font-bold">{fps} fps</span>
      </div>

      <video ref={videoRef} className="hidden" playsInline muted />

      {/* Canvas area */}
      <div className="relative z-10 mx-auto flex max-w-[1280px] flex-col gap-4 p-3 md:p-6">
        <div className="relative aspect-[16/9] w-full overflow-hidden rounded-[28px] border border-white/10 bg-black shadow-[0_20px_80px_rgba(0,0,0,0.6)] md:aspect-[16/8.5]">
          <canvas ref={videoCanvasRef} className="absolute inset-0 h-full w-full object-cover" />
          <canvas ref={drawCanvasRef} className="absolute inset-0 h-full w-full object-cover" />
          <canvas ref={cursorCanvasRef} className="absolute inset-0 h-full w-full object-cover" />

          {/* Top left — color + brush */}
          <div className="absolute left-3 top-3 flex flex-col gap-3 md:left-4 md:top-4">
            <div className="flex flex-col items-center gap-2 rounded-2xl border border-white/10 bg-black/40 px-2.5 py-3 backdrop-blur-xl">
              <span className="text-[10px] font-bold tracking-widest text-white/40">COLOR</span>
              <div className="grid grid-cols-2 gap-2">
                {COLORS.map((c) => (
                  <button
                    key={c}
                    onClick={() => setActiveColor(c)}
                    className={`h-7 w-7 rounded-full border-2 transition-all md:h-8 md:w-8 ${activeColor === c ? "scale-110 border-white shadow-lg" : "border-white/10 hover:border-white/30"}`}
                    style={{ backgroundColor: c, boxShadow: activeColor === c ? `0 0 12px ${c}80` : undefined }}
                    aria-label={`Color ${c}`}
                  />
                ))}
              </div>
              <div className="mt-1 h-px w-full bg-white/10" />
              <span className="text-[10px] font-bold tracking-widest text-white/40">SIZE</span>
              <div className="flex flex-col gap-1.5">
                {BRUSH_SIZES.map((s) => (
                  <button
                    key={s}
                    onClick={() => setBrushWidth(s)}
                    className={`flex h-7 w-7 items-center justify-center rounded-full border transition md:h-8 md:w-8 ${brushWidth === s ? "border-white bg-white text-black" : "border-white/10 bg-white/5 text-white/70 hover:bg-white/10"}`}
                  >
                    <span className="rounded-full bg-current" style={{ width: Math.min(s, 14), height: Math.min(s, 14) }} />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Top right — gesture status */}
          <div className="absolute right-3 top-3 flex flex-col items-end gap-2 md:right-4 md:top-4">
            <div className="flex items-center gap-3 rounded-2xl border border-white/10 bg-black/40 px-4 py-3 backdrop-blur-xl">
              <div className={`flex h-10 w-10 items-center justify-center rounded-xl border bg-white/5 text-lg ${meta.color} border-white/10`}>{meta.icon}</div>
              <div className="text-right leading-tight">
                <div className={`text-sm font-bold ${meta.color}`}>{meta.label}</div>
                <div className="text-xs text-white/50">{meta.sub}</div>
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

          {/* Bottom center — legend */}
          <div className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border border-white/10 bg-black/50 px-2 py-1.5 text-[11px] font-medium backdrop-blur-xl md:bottom-4 md:gap-2 md:px-4 md:text-xs">
            <span className="flex items-center gap-1.5 whitespace-nowrap">
              <span className="rounded-full bg-violet-600 px-2 py-0.5 text-[10px] font-bold text-white">PINCH</span>
              <span className="hidden text-white/80 md:inline">Draw</span>
            </span>
            <span className="h-4 w-px bg-white/10" />
            <span className="whitespace-nowrap text-white/70">✌️ Undo</span>
            <span className="h-4 w-px bg-white/10" />
            <span className="whitespace-nowrap text-white/70">✊ Clear</span>
          </div>

          {/* Bottom right — actions */}
          <div className="absolute bottom-3 right-3 flex items-center gap-2 md:bottom-4 md:right-4">
            <button onClick={undoLast} className="rounded-full border border-white/10 bg-black/50 px-4 py-2.5 text-xs font-bold backdrop-blur-xl transition hover:bg-white/10 md:px-5">
              ↩ Undo
            </button>
            <button onClick={clearCanvas} className="rounded-full border border-red-500/20 bg-red-600/90 px-4 py-2.5 text-xs font-bold backdrop-blur-xl transition hover:bg-red-600 md:px-5">
              Clear
            </button>
            <button onClick={saveDrawing} className="rounded-full bg-white px-5 py-2.5 text-xs font-bold text-black shadow-lg transition hover:bg-zinc-100">
              ⬇ Save
            </button>
          </div>

          {/* Loading / error overlay */}
          {(!isLoaded || error) && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6">
              <div className="w-full max-w-md rounded-3xl border border-white/10 bg-zinc-900 p-6 text-center shadow-2xl">
                {!error ? (
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
                    <button onClick={() => window.location.reload()} className="mt-4 rounded-full bg-white px-5 py-2 text-sm font-bold text-black">
                      Retry
                    </button>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Helper bar below canvas */}
        <div className="flex flex-col items-center justify-between gap-3 rounded-2xl border border-white/5 bg-white/[0.03] px-4 py-3 backdrop-blur md:flex-row">
          <p className="text-center text-xs leading-relaxed text-white/50 md:text-left">
            <span className="font-semibold text-white/80">Tip:</span> Hold pinch steady to draw smooth lines · Keep hand centered and well-lit · Works best in Chrome/Edge
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
