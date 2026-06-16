import os
import pickle
import pandas as pd
import numpy as np

class BallPhysicsAnalyzer:
    def __init__(self, pkl_path='tracker_stubs/ball_detections.pkl', window_size=3, video_path=None):
        self.pkl_path = pkl_path
        self.window_size = window_size
        self.video_path = video_path
        self.df = None
        self.events = {}

    def load_and_preprocess(self):
        if not os.path.exists(self.pkl_path):
            raise FileNotFoundError(f"Ball detections stub not found: {self.pkl_path}")
            
        with open(self.pkl_path, 'rb') as f:
            ball_detections = pickle.load(f)
            
        records = []
        for frame_idx, detections in enumerate(ball_detections):
            # The ball tracking dictionary stores the primary detection at key 1
            if 1 in detections:
                data = detections[1]
                if isinstance(data, dict):
                    bbox = data['bbox']
                else:
                    bbox = data
                
                # Calculate center coordinates
                x_center = (bbox[0] + bbox[2]) / 2.0
                y_center = (bbox[1] + bbox[3]) / 2.0
                records.append({
                    'frame': frame_idx,
                    'x_center': x_center,
                    'y_center': y_center
                })
            else:
                # Append NaN for frames without detections
                records.append({
                    'frame': frame_idx,
                    'x_center': np.nan,
                    'y_center': np.nan
                })
                
        df = pd.DataFrame(records)
        
        # Track raw detection status for outlier de-flickering
        detected_mask = [1 in d for d in ball_detections]
        
        # Pass 1: Remove single-frame spikes
        for t in range(1, len(df) - 1):
            if detected_mask[t]:
                # find prev detected within 5 frames
                prev_idx = None
                for p in range(t-1, max(-1, t-6), -1):
                    if detected_mask[p]:
                        prev_idx = p
                        break
                # find next detected within 5 frames
                next_idx = None
                for n in range(t+1, min(len(df), t+6)):
                    if detected_mask[n]:
                        next_idx = n
                        break
                        
                if prev_idx is not None and next_idx is not None:
                    px, py = df.iloc[prev_idx]['x_center'], df.iloc[prev_idx]['y_center']
                    cx, cy = df.iloc[t]['x_center'], df.iloc[t]['y_center']
                    nx, ny = df.iloc[next_idx]['x_center'], df.iloc[next_idx]['y_center']
                    
                    d_prev = np.sqrt((cx - px)**2 + (cy - py)**2)
                    d_next = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                    d_bridge = np.sqrt((nx - px)**2 + (ny - py)**2)
                    
                    speed_prev = d_prev / (t - prev_idx)
                    speed_next = d_next / (next_idx - t)
                    
                    if (speed_prev > 150 and speed_next > 150) or (d_prev > 150 and d_next > 150 and d_bridge < 120):
                        df.loc[t, 'x_center'] = np.nan
                        df.loc[t, 'y_center'] = np.nan
                        detected_mask[t] = False

        # Pass 2: Remove two-frame consecutive spikes
        for t in range(1, len(df) - 2):
            if detected_mask[t] and detected_mask[t+1]:
                # Find prev detected before t
                prev_idx = None
                for p in range(t-1, max(-1, t-6), -1):
                    if detected_mask[p]:
                        prev_idx = p
                        break
                # Find next detected after t+1
                next_idx = None
                for n in range(t+2, min(len(df), t+7)):
                    if detected_mask[n]:
                        next_idx = n
                        break
                        
                if prev_idx is not None and next_idx is not None:
                    px, py = df.iloc[prev_idx]['x_center'], df.iloc[prev_idx]['y_center']
                    c1x, c1y = df.iloc[t]['x_center'], df.iloc[t]['y_center']
                    c2x, c2y = df.iloc[t+1]['x_center'], df.iloc[t+1]['y_center']
                    nx, ny = df.iloc[next_idx]['x_center'], df.iloc[next_idx]['y_center']
                    
                    d_prev = np.sqrt((c1x - px)**2 + (c1y - py)**2)
                    d_next = np.sqrt((nx - c2x)**2 + (ny - c2y)**2)
                    d_bridge = np.sqrt((nx - px)**2 + (ny - py)**2)
                    
                    speed_prev = d_prev / (t - prev_idx)
                    speed_next = d_next / (next_idx - (t+1))
                    
                    if (speed_prev > 150 and speed_next > 150) or (d_prev > 150 and d_next > 150 and d_bridge < 120):
                        df.loc[t, 'x_center'] = np.nan
                        df.loc[t, 'y_center'] = np.nan
                        df.loc[t+1, 'x_center'] = np.nan
                        df.loc[t+1, 'y_center'] = np.nan
                        detected_mask[t] = False
                        detected_mask[t+1] = False
        
        # Interpolate NaNs to handle gaps
        df['x_center'] = df['x_center'].interpolate(method='linear')
        df['y_center'] = df['y_center'].interpolate(method='linear')
        df = df.bfill()
        
        # Apply a rolling window filter to smooth out pixel jitter (reduced size window to preserve dinks)
        df['x_smooth'] = df['x_center'].rolling(window=self.window_size, min_periods=1, center=True).mean()
        df['y_smooth'] = df['y_center'].rolling(window=self.window_size, min_periods=1, center=True).mean()
        
        self.df = df
        return df

    def compute_kinematics(self):
        if self.df is None:
            self.load_and_preprocess()
            
        # Velocity components (dx/dt, dy/dt)
        self.df['vx'] = self.df['x_smooth'].diff()
        self.df['vy'] = self.df['y_smooth'].diff()
        
        # Acceleration components
        self.df['ax'] = self.df['vx'].diff()
        self.df['ay'] = self.df['vy'].diff()
        
        # Magnitude of changes
        self.df['vel_magnitude'] = np.sqrt(self.df['vx']**2 + self.df['vy']**2)
        self.df['acc_magnitude'] = np.sqrt(self.df['ax']**2 + self.df['ay']**2)
        
    def detect_events(self, bounce_height_threshold=1.5, hit_acc_threshold=1.4, cooldown_frames=5):
        if self.df is None or 'vx' not in self.df.columns:
            self.compute_kinematics()
            
        self.events = {}
        n = len(self.df)
        
        # Determine frame height dynamically from video
        frame_height = 1080  # default fallback
        import cv2
        import glob
        
        # Use constructor video_path if specified
        video_path = self.video_path
        
        if not video_path or not os.path.exists(video_path):
            # Fallback scan for any video in input_videos
            video_files = []
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                video_files.extend(glob.glob(os.path.join("input_videos", ext)))
            if video_files:
                video_path = video_files[0]
                
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"[PHYSICS] Dynamically loaded frame height from {video_path}: {frame_height} px")
        else:
            print(f"[WARNING] Video file not found. Defaulting to frame height: {frame_height} px")
            
        # Load court keypoints to get the far baseline Y limit
        court_stub_path = 'tracker_stubs/court_keypoints.pkl'
        far_baseline_y = frame_height * 0.35  # Default fallback
        if os.path.exists(court_stub_path):
            try:
                with open(court_stub_path, 'rb') as f:
                    kp = pickle.load(f)
                if len(kp) >= 3:
                    far_baseline_y = min(kp[0][1], kp[1][1], kp[2][1])
                    print(f"[PHYSICS] Court keypoints loaded. Far baseline Y limit: {far_baseline_y:.1f} px")
            except Exception as e:
                print(f"[WARNING] Failed to load court keypoints for limit check: {e}")
                
        # Keep track of bounce frames to suppress adjacent hit false positives
        bounce_frames = set()
        
        # Step 1: Detect all Bounces first (Primary Priority with Spatial Height Constraint)
        # Bounces are highly distinct vertical velocity V-shaped inflections.
        for t in range(2, n - 2):
            row = self.df.iloc[t]
            frame_idx = int(row['frame'])
            
            # y-axis peak check (y(t-1) < y(t) > y(t+1) in screen space downward coordinates)
            y_curr = row['y_smooth']
            y_prev = self.df.iloc[t-1]['y_smooth']
            y_next = self.df.iloc[t+1]['y_smooth']
            
            is_y_peak = (y_prev < y_curr) and (y_curr > y_next)
            
            if is_y_peak:
                # Check both ay(t) and ay(t+1) since physical bounce deceleration peak
                # often registers on the frame immediately after the screen-space Y maximum.
                ay_t = row['ay']
                ay_next = self.df.iloc[t+1]['ay']
                min_ay = min(ay_t, ay_next)
                
                if min_ay < -bounce_height_threshold:
                    # Bounces must occur below the far baseline (with a 15px buffer for safety)
                    if y_curr > (far_baseline_y - 15):
                        bounce_frames.add(frame_idx)
                    else:
                        # Trajectory inversion too high in the air
                        pass
        
        # Step 2: Resolve Events with Priority Hierarchy, Cooldown Lockouts, and Mutual Exclusivity
        cooldown_counter = 0
        
        for t in range(2, n - 2):
            # Decrement cooldown counter if active
            if cooldown_counter > 0:
                cooldown_counter -= 1
                continue
                
            row = self.df.iloc[t]
            frame_idx = int(row['frame'])
            
            # --- Priority 1: Court Floor Bounce ---
            if frame_idx in bounce_frames:
                self.events[frame_idx] = "BOUNCE"
                cooldown_counter = cooldown_frames  # Start temporal lockout
                continue
                
            # --- Priority 2: Paddle Hit ---
            # Mutual Exclusivity: Ignore hits if within 2 frames of a registered bounce
            is_near_bounce = any(abs(frame_idx - bf) <= 2 for bf in bounce_frames)
            if is_near_bounce:
                continue
                
            vx_curr = row['vx']
            vx_prev = self.df.iloc[t-1]['vx']
            vy_curr = row['vy']
            vy_prev = self.df.iloc[t-1]['vy']
            
            delta_vx = abs(vx_curr - vx_prev)
            delta_vy = abs(vy_curr - vy_prev)
            
            # Sensitive direction sign change (minimum velocity checkpoint lowered to 0.05 to capture soft dinks)
            dir_change_x = (np.sign(vx_curr) != np.sign(vx_prev)) and (abs(vx_curr) > 0.05 and abs(vx_prev) > 0.05)
            dir_change_y = (np.sign(vy_curr) != np.sign(vy_prev)) and (abs(vy_curr) > 0.05 and abs(vy_prev) > 0.05)
            
            # Reversal acceleration thresholds lowered from 1.0 to 0.5 (50% reduction) to catch soft blocks
            if (delta_vx > hit_acc_threshold) or (delta_vy > hit_acc_threshold) or \
               ((dir_change_x or dir_change_y) and (abs(row['ax']) > 0.5 or abs(row['ay']) > 0.5)):
                self.events[frame_idx] = "HIT"
                cooldown_counter = cooldown_frames  # Start temporal lockout
 
        return self.events

    def save_events(self, output_path='tracker_stubs/ball_events.pkl'):
        # Ensure stub directory exists
        stub_dir = os.path.dirname(output_path)
        if stub_dir and not os.path.exists(stub_dir):
            os.makedirs(stub_dir)
            
        with open(output_path, 'wb') as f:
            pickle.dump(self.events, f)
        print(f"Successfully saved {len(self.events)} ball events to: {output_path}")

    def print_summary(self):
        print("\n=== Ball Tracking Physics Event Logs ===")
        print(f"{'Frame':<8} | {'Event Type':<10} | {'X Coord':<10} | {'Y Coord':<10}")
        print("-" * 46)
        for frame, event in sorted(self.events.items()):
            row = self.df[self.df['frame'] == frame].iloc[0]
            print(f"{frame:<8} | {event:<10} | {row['x_center']:<10.2f} | {row['y_center']:<10.2f}")
        print("========================================")

if __name__ == "__main__":
    analyzer = BallPhysicsAnalyzer()
    analyzer.load_and_preprocess()
    analyzer.compute_kinematics()
    analyzer.detect_events()
    analyzer.print_summary()
    analyzer.save_events()
