import { useEffect, useRef, useState } from 'react';
import { Play, Pause, Settings, Volume2, RotateCcw, Maximize2 } from 'lucide-react';

interface BBox {
  bbox: [number, number, number, number];
  is_on_court: boolean;
}

interface PlayerDetections {
  [frameIdx: string]: {
    [trackId: string]: BBox;
  };
}

interface BallDetections {
  [trackId: string]: [number, number, number, number];
}

interface BallEvents {
  [frameIdx: string]: 'HIT' | 'BOUNCE';
}

type Keypoint = [number, number];

export default function App() {
  // Video and Canvas References
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Loaded Data States
  const [courtKeypoints, setCourtKeypoints] = useState<Keypoint[] | null>(null);
  const [playerDetections, setPlayerDetections] = useState<PlayerDetections | null>(null);
  const [ballDetections, setBallDetections] = useState<BallDetections[] | null>(null);
  const [ballEvents, setBallEvents] = useState<BallEvents | null>(null);

  // Playback Control States
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isMuted, setIsMuted] = useState(true);

  // Interactive Layer Configurations
  const [showBallTrail, setShowBallTrail] = useState(true);
  const [showPlayers, setShowPlayers] = useState(true);
  const [showBounces, setShowBounces] = useState(true);
  const [showCourtOutline, setShowCourtOutline] = useState(true);
  const [showTimeFrames, setShowTimeFrames] = useState(true);

  // Settings dropdown visibility
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);

  // Fetch all JSON data on mount
  useEffect(() => {
    Promise.all([
      fetch('/data/court_keypoints.json').then((res) => res.json()),
      fetch('/data/player_detections.json').then((res) => res.json()),
      fetch('/data/ball_detections.json').then((res) => res.json()),
      fetch('/data/ball_events.json').then((res) => res.json()),
    ])
      .then(([kp, players, ball, events]) => {
        setCourtKeypoints(kp);
        setPlayerDetections(players);
        setBallDetections(ball);
        setBallEvents(events);
        setTotalFrames(players.length);
        console.log('[DASHBOARD] Loaded data assets successfully.');
      })
      .catch((err) => console.error('[ERROR] Failed to load JSON data stubs:', err));
  }, []);

  // Sync Video time updates
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      if (duration > 0 && totalFrames > 0) {
        const fps = totalFrames / duration;
        const frame = Math.min(
          Math.floor(videoRef.current.currentTime * fps),
          totalFrames - 1
        );
        setCurrentFrame(frame);
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  // Play/Pause toggler
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Video Speed modifier
  const handleSetSpeed = (rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) {
      videoRef.current.playbackRate = rate;
    }
  };

  // Drag and scrub progress timeline
  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetVal = parseFloat(e.target.value);
    setCurrentTime(targetVal);
    if (videoRef.current && duration > 0 && totalFrames > 0) {
      videoRef.current.currentTime = targetVal;
      const fps = totalFrames / duration;
      const frame = Math.min(Math.floor(targetVal * fps), totalFrames - 1);
      setCurrentFrame(frame);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const resetPlayback = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      setCurrentTime(0);
      setCurrentFrame(0);
      if (isPlaying) {
        videoRef.current.play();
      }
    }
  };

  const toggleFullscreen = () => {
    if (containerRef.current) {
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen().catch((err) => {
          console.error(`[ERROR] Failed to enable fullscreen: ${err.message}`);
        });
      } else {
        document.exitFullscreen();
      }
    }
  };

  // Master Canvas Drawing Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !playerDetections || !ballDetections) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;

    const render = () => {
      // Clear previous frames
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Court & Net Outline
      if (showCourtOutline && courtKeypoints && courtKeypoints.length === 12) {
        ctx.strokeStyle = 'rgba(253, 224, 71, 0.9)'; // Bright Yellow
        ctx.lineWidth = 2.5;
        ctx.setLineDash([]);

        const kp = courtKeypoints;

        const drawLine = (ptA: Keypoint, ptB: Keypoint) => {
          ctx.beginPath();
          ctx.moveTo(ptA[0], ptA[1]);
          ctx.lineTo(ptB[0], ptB[1]);
          ctx.stroke();
        };

        // Draw boundaries (Outer box)
        drawLine(kp[0], kp[2]);
        drawLine(kp[2], kp[11]);
        drawLine(kp[11], kp[9]);
        drawLine(kp[9], kp[0]);

        // Kitchen Lines (horizontal lines at 7ft from net)
        drawLine(kp[3], kp[5]); // Far kitchen
        drawLine(kp[6], kp[8]); // Near kitchen

        // Center lines
        drawLine(kp[1], kp[4]); // Far center
        drawLine(kp[7], kp[10]); // Near center

        // Net posts (computed perspective midpoints between kitchen lines)
        const netLeft: Keypoint = [(kp[3][0] + kp[6][0]) / 2, (kp[3][1] + kp[6][1]) / 2];
        const netRight: Keypoint = [(kp[5][0] + kp[8][0]) / 2, (kp[5][1] + kp[8][1]) / 2];
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'; // White Net Highlight to distinguish it from the yellow court
        ctx.lineWidth = 3;
        drawLine(netLeft, netRight);
      }

      // 2. Draw Bounces (Ripple Effect)
      if (showBounces && ballEvents) {
        const bounceFrames = Object.keys(ballEvents)
          .filter((f) => ballEvents[f] === 'BOUNCE')
          .map(Number)
          .sort((a, b) => a - b);

        bounceFrames.forEach((bf) => {
          const diff = currentFrame - bf;
          // Render bounce ripple animation if within +24 frames after bounce
          if (diff >= 0 && diff < 24) {
            const ballBox = ballDetections[bf]?.['1'];
            if (ballBox) {
              const bx = (ballBox[0] + ballBox[2]) / 2;
              const by = ballBox[3]; // Bottom center of ball at bounce point
              
              const progress = diff / 24;
              const maxRadius = 50;
              const radius = progress * maxRadius;
              const opacity = 1.0 - progress;

              ctx.strokeStyle = `rgba(234, 179, 8, ${opacity})`;
              ctx.lineWidth = 2;
              ctx.beginPath();
              // Flatten ellipse representing perspective surface bounce
              ctx.ellipse(bx, by, radius, radius * 0.35, 0, 0, 2 * Math.PI);
              ctx.stroke();
            }
          }
        });
      }

      // 3. Draw Players (Ground Ellipse Rings & Head Tags)
      if (showPlayers && playerDetections) {
        const framePlayers = playerDetections[currentFrame] || {};
        
        // Broadcast Color Mapping matching PBSort tracker
        const colorMap: { [key: number]: string } = {
          1: 'rgba(255, 95, 0, 1)',   // Player 1: Neon Orange
          2: 'rgba(255, 215, 0, 1)',  // Player 2: Yellow/Gold
          3: 'rgba(0, 191, 255, 1)',  // Player 3: Cyan/Light Blue
          4: 'rgba(79, 70, 229, 1)'   // Player 4: Royal Indigo
        };

        Object.entries(framePlayers).forEach(([pidStr, data]) => {
          const pid = parseInt(pidStr);
          if (!data || !data.bbox) return;

          const [x1, y1, x2, y2] = data.bbox;
          const color = colorMap[pid] || 'rgba(59, 130, 246, 1)';

          const centerX = (x1 + x2) / 2;
          const centerY = y2;
          const radiusX = (x2 - x1) * 0.65;
          const radiusY = radiusX * 0.3;

          // Draw semi-transparent floor footprint ring
          ctx.fillStyle = color.replace('1)', '0.25)'); // Fill with 25% opacity
          ctx.beginPath();
          ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
          ctx.fill();

          // Stroke ground ellipse
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
          ctx.stroke();

          // Centered Header Pill Tag
          const label = `Player ${pid}`;
          ctx.font = '600 11px Inter, sans-serif';
          const textWidth = ctx.measureText(label).width;
          const tagY = Math.max(y1 - 14, 20);

          // Draw Pill Body
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.roundRect(centerX - textWidth/2 - 8, tagY - 14, textWidth + 16, 18, 9);
          ctx.fill();

          // Draw Text
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, centerX - textWidth/2, tagY - 1);
        });
      }

      // 4. Draw Ball Trajectory Arc (Last Bounce to Next Bounce)
      if (showBallTrail && ballEvents && ballDetections) {
        const bounceFrames = Object.keys(ballEvents)
          .filter((f) => ballEvents[f] === 'BOUNCE')
          .map(Number)
          .sort((a, b) => a - b);

        const lastBounce = bounceFrames.filter((bf) => bf <= currentFrame).pop() || 0;
        const nextBounce = bounceFrames.find((bf) => bf > currentFrame) || totalFrames - 1;

        // Trace points from lastBounce to currentFrame
        const trajectoryPoints: { x: number; y: number }[] = [];
        for (let f = lastBounce; f <= currentFrame; f++) {
          const ballBox = ballDetections[f]?.['1'];
          if (ballBox) {
            trajectoryPoints.push({
              x: (ballBox[0] + ballBox[2]) / 2,
              y: (ballBox[1] + ballBox[3]) / 2
            });
          }
        }

        if (trajectoryPoints.length > 1) {
          ctx.lineWidth = 4;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          for (let idx = 1; idx < trajectoryPoints.length; idx++) {
            const progress = idx / trajectoryPoints.length;
            ctx.strokeStyle = `rgba(34, 197, 94, ${progress})`; // Glow green fading trail
            ctx.beginPath();
            ctx.moveTo(trajectoryPoints[idx - 1].x, trajectoryPoints[idx - 1].y);
            ctx.lineTo(trajectoryPoints[idx].x, trajectoryPoints[idx].y);
            ctx.stroke();
          }

          // Draw current ball center dot
          const lastPoint = trajectoryPoints[trajectoryPoints.length - 1];
          ctx.fillStyle = 'rgba(34, 197, 94, 1)';
          ctx.shadowColor = 'rgba(34, 197, 94, 0.8)';
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(lastPoint.x, lastPoint.y, 5, 0, 2 * Math.PI);
          ctx.fill();
          // Reset shadow
          ctx.shadowBlur = 0;
        }
      }

      // 5. Draw Time & Frames Overlay
      if (showTimeFrames) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        ctx.roundRect(15, 15, 230, 48, 8);
        ctx.fill();

        ctx.font = '500 11px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`Frame: `, 25, 33);
        ctx.fillText(`Time: `, 25, 51);

        ctx.font = 'bold 11px monospace';
        ctx.fillStyle = '#f8fafc';
        ctx.fillText(`${currentFrame} / ${totalFrames - 1}`, 70, 33);
        ctx.fillText(`${(currentTime * 1000).toFixed(0)} ms`, 70, 51);
      }

      animationId = requestAnimationFrame(render);
    };

    // Begin looping
    render();

    return () => cancelAnimationFrame(animationId);
  }, [
    currentFrame,
    courtKeypoints,
    playerDetections,
    ballDetections,
    ballEvents,
    showBallTrail,
    showPlayers,
    showBounces,
    showCourtOutline,
    showTimeFrames,
  ]);

  // Adjust canvas size according to loaded video dimensions
  const handleResize = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  };

  return (
    <div className="app-container" ref={containerRef}>
      {/* Header bar */}
      <div className="header-bar">
        <div className="title-container">
          <h1>INBOUND VISION</h1>
          <p>Pickleball Match Player Analysis</p>
        </div>
        <div className="badge-active">
          <span className="pulse-dot"></span>
          <span>Analysis Active</span>
        </div>
      </div>

      {/* Main player screen layout */}
      <div className="layout-grid">
        
        {/* Left Video viewport */}
        <div className="video-card">
          <div className="video-wrapper">
            {/* Native Video Player */}
            <video
              ref={videoRef}
              src="/video.mp4"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onLoadedData={handleResize}
              onClick={togglePlay}
              muted={isMuted}
              playsInline
            />

            {/* Overlaid drawing canvas */}
            <canvas
              ref={canvasRef}
              className="overlay-canvas"
            />

            {/* Floating Layer Controls (Top Right Overlay) */}
            {isSettingsOpen && (
              <div className="settings-overlay">
                <div className="settings-header">
                  <span>Layer Toggles</span>
                  <Settings className="w-3.5 h-3.5" />
                </div>
                <div className="toggle-list">
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={showBallTrail}
                      onChange={(e) => setShowBallTrail(e.target.checked)}
                    />
                    Ball (Bounce Trail)
                  </label>
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={showPlayers}
                      onChange={(e) => setShowPlayers(e.target.checked)}
                    />
                    Players (Footprint Rings)
                  </label>
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={showBounces}
                      onChange={(e) => setShowBounces(e.target.checked)}
                    />
                    Bounces (Ripple Effect)
                  </label>
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={showCourtOutline}
                      onChange={(e) => setShowCourtOutline(e.target.checked)}
                    />
                    Court & Net Outline
                  </label>
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={showTimeFrames}
                      onChange={(e) => setShowTimeFrames(e.target.checked)}
                    />
                    Time & Frames HUD
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Controls bar */}
          <div className="control-bar">
            {/* Timeline Scrub */}
            <div className="timeline-container">
              <span className="timeline-time">0:00</span>
              <input
                type="range"
                min="0"
                max={duration || 100}
                step="0.01"
                value={currentTime}
                onChange={handleScrub}
                className="scrub-timeline"
              />
              <span className="timeline-time">
                {duration ? `${Math.floor(duration / 60)}:${Math.floor(duration % 60).toString().padStart(2, '0')}` : '0:00'}
              </span>
            </div>

            {/* Playback action items */}
            <div className="actions-row">
              <div className="actions-left">
                <button
                  onClick={togglePlay}
                  className="btn-icon btn-play"
                  title={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={resetPlayback}
                  className="btn-icon"
                  title="Rewind to start"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={toggleMute}
                  className="btn-icon"
                  title={isMuted ? 'Unmute' : 'Mute'}
                >
                  <Volume2 className={`w-4 h-4 ${isMuted ? 'text-slate-500' : 'text-sky-500'}`} />
                </button>
              </div>

              {/* Advanced info overlays */}
              <div className="actions-right">
                {/* Playback speed selector group */}
                <div style={{ display: 'flex', gap: '4px', backgroundColor: '#151922', padding: '2px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                  {[0.25, 0.5, 0.75, 1].map((rate) => (
                    <button
                      key={rate}
                      onClick={() => handleSetSpeed(rate)}
                      className={`btn-text ${playbackRate === rate ? 'btn-settings-active' : ''}`}
                      style={{ padding: '4px 8px', fontSize: '10px', border: 'none', background: playbackRate === rate ? '' : 'transparent' }}
                    >
                      {rate}x
                    </button>
                  ))}
                </div>

                {/* Settings panel toggle */}
                <button
                  onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                  className={`btn-icon ${isSettingsOpen ? 'btn-settings-active' : ''}`}
                  title="Toggle Settings Panel"
                >
                  <Settings className="w-4 h-4" />
                </button>

                {/* Viewmode triggers */}
                <button
                  onClick={toggleFullscreen}
                  className="btn-icon"
                  title="Fullscreen"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Info Panels */}
        <div className="side-panel">
          
          {/* Active Players Legend panel */}
          <div className="legend-card">
            <h3 className="legend-title">Active Players</h3>
            <div className="legend-list">
              <div className="legend-item">
                <div className="legend-item-left">
                  <span className="legend-color-dot" style={{ backgroundColor: 'rgba(255, 95, 0, 1)' }}></span>
                  <span className="legend-name">Player 1</span>
                </div>
                <span className="legend-position">Near-Left</span>
              </div>
              <div className="legend-item">
                <div className="legend-item-left">
                  <span className="legend-color-dot" style={{ backgroundColor: 'rgba(255, 215, 0, 1)' }}></span>
                  <span className="legend-name">Player 2</span>
                </div>
                <span className="legend-position">Near-Right</span>
              </div>
              <div className="legend-item">
                <div className="legend-item-left">
                  <span className="legend-color-dot" style={{ backgroundColor: 'rgba(0, 191, 255, 1)' }}></span>
                  <span className="legend-name">Player 3</span>
                </div>
                <span className="legend-position">Far-Left</span>
              </div>
              <div className="legend-item">
                <div className="legend-item-left">
                  <span className="legend-color-dot" style={{ backgroundColor: 'rgba(79, 70, 229, 1)' }}></span>
                  <span className="legend-name">Player 4</span>
                </div>
                <span className="legend-position">Far-Right</span>
              </div>
            </div>
          </div>

          {/* Quick Shortcuts / Instructions Panel */}
          <div className="info-card">
            <h3 className="legend-title">Dashboard Info</h3>
            <ul className="info-list">
              <li className="info-item">
                <span className="info-bullet">•</span>
                <span><strong>Toggle Layers</strong>: Turn visual trails, footprint rings, and court boundaries on/off instantly.</span>
              </li>
              <li className="info-item">
                <span className="info-bullet">•</span>
                <span><strong>Playback Speeds</strong>: Slow down to 0.5x or 0.25x to inspect fast exchanges or line bounces frame-by-frame.</span>
              </li>
              <li className="info-item">
                <span className="info-bullet">•</span>
                <span><strong>Frame Scrubbing</strong>: Pause the video and drag the timeline track to inspect specific frame coordinates on the canvas.</span>
              </li>
            </ul>
          </div>
          
        </div>
      </div>
    </div>
  );
}
