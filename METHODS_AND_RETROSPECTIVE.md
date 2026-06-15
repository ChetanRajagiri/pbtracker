# 🏓 PBTracker — Methods, Models, and Pipeline Retrospective

This document outlines the software components, deep learning models, geometric and kinematic methods, historical attempts (both failed and successful), and architectural decisions implemented in **PBTracker**.

---

## 🛠 1. Current Technology Stack

The pipeline uses a hybrid architecture combining deep learning object detection, multi-object tracking (MOT), re-identification (ReID), perspective projection, and post-processing heuristics.

| Pipeline Component | Software/Framework | Specific Model / Algorithm |
| :--- | :--- | :--- |
| **Player Detection** | Ultralytics YOLOv8 | `yolov8x.pt` (Pre-trained Person Class) |
| **Ball Detection** | Ultralytics YOLOv5 | Custom Fine-Tuned Model (`best.pt` / `models/yolo5_last.pt`) |
| **Multi-Player Tracking** | BoxMOT / PyTorch | BoT-SORT (Appearance + Motion-based Tracker) |
| **Appearance Embeddings** | Torchvision / BoxMOT | OSNet (`osnet_x0_25_msmt17.pt`) - 512-D visual embeddings |
| **Court Mapping** | OpenCV GUI / Geometry | Manual 12-point selector + Homography projection |
| **Ball Interpolation** | Pandas | `DataFrame.interpolate(method='linear')` |
| **Post-hoc Hist Healing** | OpenCV / Numpy | HSV Color Histogram Correlation (`cv2.HISTCMP_CORREL`) |
| **Post-hoc Deep Re-ID** | PyTorch / torchvision | Headless ResNet50 (2048-D visual embeddings) |
| **Analytics Dashboard** | Streamlit & Plotly | Interactive Web App & SVG/Canvas Heatmaps |
| **Environment Sync** | Astral `uv` | Locked Python 3.12+ project virtual environment |

---

## 📐 2. Key Methods & Implementations

### A. Locked Gallery Seeding (Player ID Calibration)
To ensure players map to deterministic, standardized IDs:
1. **Observation Window:** During the first 30–150 frames, the tracker tracks all bounding boxes inside the court polygon.
2. **Stability Gate:** A raw track ID must be observed inside the court polygon for at least 4 consecutive frames to qualify.
3. **Master Assignment:** Once 4 stable tracks are found, they are sorted by spatial quadrants and locked into four Master IDs:
   * **ID 1:** Near-Left Player
   * **ID 2:** Near-Right Player
   * **ID 3:** Far-Left Player
   * **ID 4:** Far-Right Player
4. **Embedding Profiles:** For each Master ID, an average L2-normalized 512-dimensional OSNet visual embedding is computed.
5. **Dynamic Matching:** For all subsequent frames, new player detections are mapped to Master IDs 1–4 using Cosine Similarity against these locked profiles.

### B. Spectator & Sideline Filtering
To prevent spectators, refs, or sideline coaches from stealing active player IDs:
1. **Asymmetric Polygon Expansion:** The 12 court keypoints form a polygon. Camera perspective distortion (foreshortening) means players behind the far baseline are compressed into fewer vertical pixels, while players at the near baseline occupy more space. The court polygon is expanded asymmetrically:
   * Far Baseline vertices (Y <= 55% of height): pushed outward by **30px**.
   * Near Baseline vertices (Y > 55% of height): pushed outward by **80px**.
   * Sidelines: pushed outward by **40px**.
2. **3-Point Foot Check:** A player is considered on-court if any of the three foot-level points—bottom-left `(x1, y2)`, bottom-center `((x1+x2)/2, y2)`, or bottom-right `(x2, y2)`—falls inside the expanded polygon.
3. **Out-of-Court Suppression:** Detections outside the boundary bypass the OSNet embedding extractor and are assigned a negative tracking ID (mapped to ignore ID 0).

### C. Ball Physics & Officiating Engine
1. **Bounce Kinematics:** Bounces are flagged by monitoring vertical pixel coordinates for directional velocity shifts (peaks/valleys in Y-coordinate) combined with deceleration/acceleration threshold gates.
2. **Floor Constraint:** Bounces are restricted to the lower half of the frame to prevent overhead swings or net-touches from flagging false bounces.
3. **Homography Projection:** The bottom-center of the ball bbox `(x, y)` at a bounce frame is projected onto the 2D top-down mini-court coordinate system (20x44 ft) using a homography matrix computed from the 12 court keypoints.
4. **Automated Line Calls:** If the projected bounce falls within $[0, 20]$ ft width and $[0, 44]$ ft length, it is classified as **IN** (Green banner overlay); otherwise, it is flagged as **OUT** (Red banner overlay).

### D. Post-Processing Track Healing
* **Tracklet Merger (HSV Histogram):** For minor tracking dropouts (e.g. crossing players), it merges track segments if their temporal gap is < 45 frames, centroid distance is < 150px, and HSV color histograms correlate above 0.85.
* **Deep Re-ID Healer (ResNet50):** For large temporal gaps, a headless ResNet50 extracts 2048-D visual feature vectors from player crops. Tracklets are clustered and re-mapped using Cosine Similarity against Master Profiles.

---

## 📈 3. Retrospective: What We Tried, Failed, and Succeeded

| Attempted Solution | Status | What Failed / Why | What Succeeded / Why |
| :--- | :--- | :--- | :--- |
| **Stateful Proximity & Quadrant Sorting** | **FAILED** | Position-based tracking failed when players switched court halves, crossed paths during rallies, or suffered brief occlusions, resulting in constant ID swapping. | **Replaced with BoT-SORT + OSNet ReID.** Kalman motion estimation plus visual feature extraction established stable identities. |
| **Pure Frame-by-Frame ReID Matching** | **PARTIALLY FAILED** | Background spectators, sideline coaches, and officials would frequently trigger the person detector, match against master profiles, and steal active player IDs. | **Implemented Locked Seeding & Sideline Suppression.** Seeding locks the 4 active players early, and out-of-court bounding boxes bypass ReID completely. |
| **Symmetric Court Boundary Expansion** | **FAILED** | Standard uniform scaling of the court boundary cut off players standing deep behind the near/far baseline due to perspective foreshortening. | **Asymmetric Perspective Expansion & 3-Point Foot Check.** Adjusted boundaries based on perspective depth and checked three foot contact points instead of a single centroid. |
| **Default BoT-SORT Tracking Buffer** | **FAILED** | The default 30-frame track buffer deleted lost tracks after 1 second, causing players to get assigned new raw IDs during brief out-of-frame movements. | **Increased Track Buffer to 90 Frames.** Keeping lost tracks alive for 3 seconds allowed Kalman state recovery when players returned. |
| **Heuristics-Only Ball Bounce Detection** | **FAILED** | Simple vertical peak detection registered false bounces on high racquet swings, overhead smashes, and fast rallies. | **Kinematic Gates & Floor Constraint.** Added minimum velocity/acceleration thresholds and restricted bounce candidates to the lower 50% of the frame. |

---

## 📊 4. Downstream Telemetry & App Dashboard
Once tracking IDs (1–4) and ball events are stabilized and saved to stubs:
* **Locomotion Metrics:** Calculated by mapping foot positions to the mini-court model, computing total distance run (in feet), average velocity, and top sprint speeds.
* **Biomechanical Distribution:** Leverages player-to-ball relative coordinates and speed to classify shot types (forehand vs. backhand) and player positioning.
* **Streamlit UI:** Reads the finalized stubs to build interactive Plotly heatmaps showing positioning density, rally timelines, and match statistics.
