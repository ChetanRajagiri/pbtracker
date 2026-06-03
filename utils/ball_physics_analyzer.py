import os
import pickle
import pandas as pd
import numpy as np

class BallPhysicsAnalyzer:
    def __init__(self, pkl_path='tracker_stubs/ball_detections.pkl', window_size=2):
        self.pkl_path = pkl_path
        self.window_size = window_size
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
        
        # Interpolate NaNs to handle gaps
        df['x_center'] = df['x_center'].interpolate(method='linear')
        df['y_center'] = df['y_center'].interpolate(method='linear')
        
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
        
    def detect_events(self, bounce_height_threshold=1.5, hit_acc_threshold=2.8, cooldown_frames=5):
        if 'vx' not in self.df.columns:
            self.compute_kinematics()
            
        self.events = {}
        n = len(self.df)
        
        # Keep track of bounce frames to suppress adjacent hit false positives
        bounce_frames = set()
        
        # Step 1: Detect all Bounces first (Primary Priority)
        # Bounces are highly distinct vertical velocity V-shaped inflections.
        for t in range(2, n - 2):
            row = self.df.iloc[t]
            frame_idx = int(row['frame'])
            
            # y-axis peak check (y(t-1) < y(t) > y(t+1) in screen space downward coordinates)
            y_curr = row['y_smooth']
            y_prev = self.df.iloc[t-1]['y_smooth']
            y_next = self.df.iloc[t+1]['y_smooth']
            
            is_y_peak = (y_prev < y_curr) and (y_curr > y_next)
            
            if is_y_peak and self.df.iloc[t]['ay'] < -bounce_height_threshold:
                # Store detected bounces
                bounce_frames.add(frame_idx)
        
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
                cooldown_counter = cooldown_frames # Start temporal lockout
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
            
            # Sensitive direction sign change (minimum velocity checkpoint lowered to 0.1 to register kitchen dinks)
            dir_change_x = (np.sign(vx_curr) != np.sign(vx_prev)) and (abs(vx_curr) > 0.1 and abs(vx_prev) > 0.1)
            dir_change_y = (np.sign(vy_curr) != np.sign(vy_prev)) and (abs(vy_curr) > 0.1 and abs(vy_prev) > 0.1)
            
            if (delta_vx > hit_acc_threshold) or (delta_vy > hit_acc_threshold) or \
               ((dir_change_x or dir_change_y) and (abs(row['ax']) > 1.0 or abs(row['ay']) > 1.0)):
                self.events[frame_idx] = "HIT"
                cooldown_counter = cooldown_frames # Start temporal lockout

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
