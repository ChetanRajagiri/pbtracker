# 🏓 PBTracker — Pickleball Video Analysis Pipeline

A computer vision pipeline for **tracking players and the ball** in pickleball match footage. Built with YOLO object detection, OpenCV geometric filtering, and deep learning–based person re-identification for robust multi-player tracking.

---

## 📁 Project Structure

```
pbtracker/
├── main.py                          # Main pipeline entry point
├── pyproject.toml                   # Project config & dependencies
│
├── trackers/                        # Core tracking modules
│   ├── player_tracker.py            #   YOLO-based player detection + court polygon filtering
│   └── ball_tracker.py              #   YOLO-based ball detection + Pandas interpolation
│
├── court_line_detector/             # Court boundary detection
│   └── manual_court_selector.py     #   Interactive 12-point court keypoint selector (OpenCV GUI)
│
├── utils/                           # Post-processing & track healing
│   ├── tracklet_merger.py           #   Spatio-temporal + HSV histogram track stitching
│   └── deep_reid_healer.py          #   ResNet50-based deep re-ID for broken track recovery
│
├── debug/                           # Debugging & diagnostic scripts
│   ├── debug_players.py             #   Visualise court polygon & player containment
│   └── yolo_inference.py            #   Standalone single-frame YOLO inference test
│
├── models/                          # YOLO model weights (git-ignored)
│   └── yolo5_last.pt                #   Fine-tuned YOLOv5 ball detection model
│
├── input_videos/                    # Source video files (git-ignored)
│   └── input.mp4
│
├── output_videos/                   # Annotated output videos (git-ignored)
│   └── player_tracking_test.mp4
│
├── tracker_stubs/                   # Cached detection data (git-ignored)
│   ├── player_detections.pkl        #   Cached player bounding boxes per frame
│   ├── ball_detections.pkl          #   Cached ball bounding boxes per frame
│   └── court_keypoints.pkl          #   Cached 12 court keypoint coordinates
│
└── yolov8x.pt                       # Pre-trained YOLOv8x for person detection (git-ignored)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd pbtracker

# Install dependencies
uv sync
```

### Required Files (not tracked in git)

| File | Purpose | Source |
|---|---|---|
| `yolov8x.pt` | Person detection model | Auto-downloaded by Ultralytics on first run |
| `models/yolo5_last.pt` | Ball detection model | Custom trained — place manually |
| `input_videos/input.mp4` | Source match footage | Your video file |

---

## 🏃 Pipeline Steps

The pipeline runs in **5 sequential stages**. Each stage caches its output as a `.pkl` stub so subsequent runs skip expensive inference.

### Step 1 — Court Keypoint Selection

```
main.py → ManualCourtDetector.get_keypoints()
```

On first run, an OpenCV window pops up showing Frame 0. **Click exactly 12 points** marking the court lines (baselines, service lines, center lines). These define the court boundary polygon used to classify players as on-court or off-court.

> Cached to: `tracker_stubs/court_keypoints.pkl`

### Step 2 — Player Detection & Tracking

```
main.py → PlayerTracker.detect_frames()
```

Runs **YOLOv8x** with `model.track(persist=True)` for multi-object tracking. Each detected person gets a persistent Track ID. A polygon containment test marks each player as `is_on_court: true/false`.

> Cached to: `tracker_stubs/player_detections.pkl`

### Step 3 — Ball Detection

```
main.py → BallTracker.detect_frames()
```

Runs a **fine-tuned YOLOv5** model to detect the pickleball in each frame.

> Cached to: `tracker_stubs/ball_detections.pkl`

### Step 4 — Ball Position Interpolation

```
main.py → BallTracker.interpolate_ball_positions()
```

Uses **Pandas linear interpolation** to fill gaps where the ball wasn't detected (motion blur, occlusion). Each position is tagged as `detected` or `interpolated`.

### Step 5 — Annotation & Video Export

```
main.py → draw_bboxes() for players, ball, and court keypoints
```

Renders the final annotated video with:
- **Player bounding boxes** — green (on-court) / orange (off-court) with Track ID labels
- **Ball markers** — green box (YOLO detected) / pink box (Pandas interpolated)
- **Court keypoints** — blue numbered dots
- **Frame counter** — top-right corner

> Output: `output_videos/player_tracking_test.mp4`

---

## 🔧 Post-Processing Tools

After the main pipeline runs, use these standalone scripts to heal broken tracking IDs.

### Tracklet Merger (HSV Histogram Correlation)

Stitches broken track IDs using **spatio-temporal proximity** and **HSV color histogram matching**.

```bash
# Run with defaults (45-frame window, 150px distance, 0.85 histogram threshold)
uv run python utils/tracklet_merger.py

# Custom thresholds
uv run python utils/tracklet_merger.py --temporal 60 --spatial 200 --hist 0.80
```

| Gate | Default | Description |
|---|---|---|
| Temporal | 45 frames | Max gap between old track ending and new track starting |
| Spatial | 150 px | Max centroid distance between ending/starting bboxes |
| Visual | 0.85 | Min HSV histogram correlation (cv2.HISTCMP_CORREL) |

### Deep Re-ID Healer (ResNet50 Neural Matching)

Uses a **headless ResNet50** backbone to extract 2048-D visual embeddings and match fragment tracklets against baseline identity profiles via **cosine similarity**.

```bash
# Run with defaults (0.88 cosine threshold)
uv run python utils/deep_reid_healer.py

# Auto-strip all noise tracks, keeping only primary players
uv run python utils/deep_reid_healer.py --drop-noise

# Custom keep set
uv run python utils/deep_reid_healer.py --drop-noise --keep 1 2 3 4 5

# Manually drop specific IDs
uv run python utils/deep_reid_healer.py --drop 8 15 56

# Adjust thresholds
uv run python utils/deep_reid_healer.py --threshold 0.90 --profile-frames 200
```


## 📊 Analytics Dashboard

InBound Vision includes an interactive post-match performance dashboard built with **Streamlit** and **Plotly**. The dashboard reads the cached player tracking, ball positions, and physics events to display:
- **Interactive Heatmaps** of player movement mapped to a 2D court model.
- **Match Dynamics** (total rallies, average shots per rally, game summary).
- **Locomotion Metrics** (total distance covered, average and maximum sprint speed per player).
- **Biomechanical Stats** (forehand vs. backhand distribution, unforced net/out faults, and DUPR rating).

To run the dashboard:

```bash
uv run streamlit run app.py
```

---

## 🔄 Recommended Workflow

```bash
# 1. Run the main pipeline (first run triggers court selection GUI)
uv run python main.py

# 2. Heal fragmented track IDs with histogram matching
uv run python utils/tracklet_merger.py

# 3. Deep neural re-ID healing for large temporal gaps
uv run python utils/deep_reid_healer.py --drop-noise

# 4. Re-render the output video with healed tracking data
uv run python main.py
```

### Clearing Cache

To force a fresh re-inference (e.g., after changing models or video), delete the relevant stub:

```bash
# Clear all caches
rm tracker_stubs/*.pkl

# Clear only player tracking (forces YOLO re-run)
rm tracker_stubs/player_detections.pkl

# Clear only ball tracking
rm tracker_stubs/ball_detections.pkl

# Clear court keypoints (triggers re-selection GUI)
rm tracker_stubs/court_keypoints.pkl
```

---

## 📊 Data Formats

### Player Detections (`player_detections.pkl`)

```python
# List of dicts, one per frame. Each dict maps track_id → metadata.
[
    {                                    # Frame 0
        1: {'bbox': [x1, y1, x2, y2], 'is_on_court': True},
        2: {'bbox': [x1, y1, x2, y2], 'is_on_court': False},
    },
    { ... },                             # Frame 1
    ...
]
```

### Ball Detections (`ball_detections.pkl`)

```python
# List of dicts, one per frame. Key 1 = ball.
[
    {1: [x1, y1, x2, y2]},              # Frame 0 — detected
    {},                                   # Frame 1 — no detection
    ...
]
```

After interpolation, the format becomes:

```python
{1: {'bbox': [x1, y1, x2, y2], 'source': 'detected'}}     # or 'interpolated'
```

### Court Keypoints (`court_keypoints.pkl`)

```python
# List of 12 (x, y) tuples defining the court lines.
[(x0, y0), (x1, y1), ..., (x11, y11)]
```

The court boundary polygon for filtering uses corners at indices `[0, 2, 11, 9]`.

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Object Detection | [Ultralytics YOLO](https://docs.ultralytics.com/) (v8 for players, v5 for ball) |
| Multi-Object Tracking | YOLO BoT-SORT (`model.track(persist=True)`) |
| Court Detection | Manual 12-point OpenCV GUI selector |
| Ball Interpolation | Pandas `DataFrame.interpolate(method='linear')` |
| Track Healing | HSV histogram correlation (OpenCV) + ResNet50 cosine similarity (PyTorch) |
| Video I/O | OpenCV `VideoCapture` / `VideoWriter` |

---

## 📝 License

This project is for educational and research purposes.
