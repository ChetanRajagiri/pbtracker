# 🏓 PBTracker — Pickleball Video Analysis Pipeline

A computer vision pipeline for **tracking players and the ball** in pickleball match footage. Built with YOLO object detection, OpenCV geometric filtering, and deep learning–based person re-identification for robust multi-player tracking.

---

## 📁 Project Structure

```
pbtracker/
├── main.py                          # Main pipeline entry point (Detection + ReID Filter + Ball Tracker)
├── pyproject.toml                   # Project config & dependencies
│
├── trackers/                        # Core tracking modules
│   ├── player_tracker.py            #   YOLO-based tiled player detection + court filtering
│   └── ball_tracker.py              #   YOLO-based ball detection + Pandas interpolation
│
├── court_line_detector/             # Court boundary detection
│   └── manual_court_selector.py     #   Interactive 12-point court keypoint selector (OpenCV GUI)
│
├── utils/                           # Post-processing & diagnostics
│   ├── auto_player_filter.py        #   Identity-first ReID matching, spatial tie-breaker & fallbacks
│   ├── debug_missing_box.py         #   Trace missing bboxes and similarity scores for raw tracks
│   └── extract_crossover_frames.py  #   Identify crossover frames for verification
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
git clone https://github.com/ChetanRajagiri/pbtracker
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

The pipeline runs in **4 sequential stages**. Each stage caches its output as a `.pkl` stub so subsequent runs skip expensive inference.

### Step 1 — Court Keypoint Selection

```
main.py → ManualCourtDetector.get_keypoints()
```

On first run, an OpenCV window pops up showing Frame 0. **Click exactly 12 points** marking the court lines.

> Cached to: `tracker_stubs/court_keypoints.pkl`

### Step 2 — Tiled Player Detection & Tracking

```
main.py → PlayerTracker.detect_frames()
```

Runs **YOLOv8x** using a horizontal tiled/cropped approach (separating the far-court and near-court regions to maintain resolution for small players). Outputs are merged and deduplicated using IoU.

> Cached to: `tracker_stubs/player_detections.pkl`

### Step 3 — Appearance-Based Player Filtration

```
main.py → AutoPlayerFilter.run_filtration()
```

Performs:
1. **Gallery Seeding:** Seeds a locked appearance profile (OSNet embeddings) for all 4 players during the first stable frames.
2. **Identity-First ReID Matching:** Matches subsequent detections directly against the locked templates (threshold > 0.35).
3. **Spatial Tie-Breaker:** Resolves matching ambiguities within a 0.05 margin using spatial continuity (closest distance to last known coordinate).
4. **Positional Fallback:** Automatically assigns unmatched in-court detections to vacant slots on their respective court half.

### Step 4 — Ball Detection & Interpolation

```
main.py → BallTracker.detect_frames()
```

Runs a **fine-tuned YOLOv5** model to detect the pickleball, and fills gaps using **Pandas linear interpolation** to resolve fast-motion blur and occlusions.

### Step 5 — Annotation & Video Export

Renders the final annotated video with:
- **Player bounding boxes** — Active player Master IDs `[1, 2, 3, 4]` labeled cleanly.
- **Ball markers** — green box (YOLO detected) / pink box (Pandas interpolated).
- **Court keypoints & Net boundary** — blue numbered dots and horizontal net split lines.

> Output: `output_videos/player_tracking_test.mp4`

---

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

Everything runs end-to-end automatically:

```bash
# Run the entire pipeline
uv run python main.py
```

### Clearing Cache

To force a fresh re-inference (e.g., after changing models or video), delete the relevant stub:

```bash
# Clear all caches
rm tracker_stubs/*.pkl

# Clear only player tracking (forces YOLO re-run)
rm tracker_stubs/player_detections.pkl

# Clear raw pre-filtration detections
rm tracker_stubs/player_detections_raw.pkl

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
