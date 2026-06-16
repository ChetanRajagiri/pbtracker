# 🏓 InBound Vision — Interactive Video Player Dashboard

A premium React + TypeScript + Vite frontend application designed for analyzing and visualizing pickleball matches with interactive canvas-based tracking layers.

---

## 🛠 Features & Layers

The dashboard overlays analytics data dynamically on top of the raw match video using an interactive HTML5 `<canvas>` synced frame-by-frame:

1. **Ball (Bounce Trail)**: Renders a glowing neon green arc representing the trajectory of the ball starting from its most recent bounce up to the next bounce event, fading historical coordinates smoothly.
2. **Players (Footprint Rings)**: Draws perspective-aligned floor footprint ellipses (styled with semi-transparent color fills matching the Indian Pickleball League broadcast scheme) at the feet of the four active players, along with named label tags centered above their heads.
3. **Bounces (Ripple Effect)**: Renders concentric expanding and fading white ripple rings at the exact coordinate where a bounce occurs for 24 frames after the event.
4. **Court & Net Outline**: Connects the 12 manual calibration keypoints with bright yellow lines to display baseline boundary limits and kitchen lines, with the net line highlighted in a clean white outline.
5. **HUD Stats**: Shows a dedicated overlay displaying current playback metrics (current Frame vs Total Frames, and timestamp in milliseconds).

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **Node.js** and **npm** installed on your system.

### How to Run Locally

1. **Export JSON Data from Pipeline stubs**:
   First, make sure to generate the necessary data feeds from your Python tracking stubs. Run this from the root directory:
   ```bash
   uv run python utils/export_json.py
   ```
   *This exports `court_keypoints.json`, `player_detections.json`, `ball_detections.json`, and `ball_events.json` directly to the `public/data/` folder.*

2. **Navigate and Run Dev Server**:
   Navigate into the dashboard directory and spin up Vite:
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

3. **Open Browser**:
   Click on the local URL printed in the terminal (typically [http://localhost:5173/](http://localhost:5173/)) to open the dashboard.

---

## ⚙️ Interactive Controls

* **Layer Toggles**: Click the floating settings panel (top right overlay) to enable or disable specific layers dynamically.
* **Playback Speeds**: Instantly switch the video speed between `0.25x`, `0.5x`, `0.75x`, and `1.0x` using the buttons on the control toolbar.
* **Frame Scrubbing**: Drag the progress timeline to seek to specific playback offsets and inspect coordinate states instantly on the canvas.
