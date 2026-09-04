import fs from "fs";
import path from "path";
import https from "https";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const DEST = path.join(process.cwd(), "public", "models", "hand_landmarker.task");

if (fs.existsSync(DEST) && fs.statSync(DEST).size > 1_000_000) {
  console.log("✓ Model already present:", (fs.statSync(DEST).size / 1024 / 1024).toFixed(2), "MB");
  process.exit(0);
}

console.log("⬇ Downloading hand_landmarker.task (7.8 MB)...");
fs.mkdirSync(path.dirname(DEST), { recursive: true });

const file = fs.createWriteStream(DEST);
https
  .get(MODEL_URL, (res) => {
    if (res.statusCode !== 200) {
      console.error("Failed to download:", res.statusCode, res.statusMessage);
      process.exit(1);
    }
    const total = parseInt(res.headers["content-length"] || "0", 10);
    let received = 0;
    res.on("data", (c) => {
      received += c.length;
      if (total) process.stdout.write(`\r  ${(received / 1024 / 1024).toFixed(2)} / ${(total / 1024 / 1024).toFixed(2)} MB`);
    });
    res.pipe(file);
    file.on("finish", () => {
      file.close(() => {
        console.log(`\n✓ Saved to ${DEST} (${(fs.statSync(DEST).size / 1024 / 1024).toFixed(2)} MB)`);
      });
    });
  })
  .on("error", (err) => {
    fs.unlink(DEST, () => {});
    console.error("Download error:", err.message);
    process.exit(1);
  });
