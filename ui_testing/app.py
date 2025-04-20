#!/usr/bin/env python3
"""
Mock Cat Laser UI Testing Application

This Flask application provides a simulated version of the cat laser tracking interface
for UI/UX testing on a Mac without requiring a Raspberry Pi or any hardware connections.
"""

import os
import time
import json
import random
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, request, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Mock system state
mock_state = {
    'tracking_active': False,
    'laser_state': 'off',
    'servo_positions': {'pan': 90, 'tilt': 90},
    'last_detection': None,
    'current_test_mode': None,
    'test_thread': None
}

# Mock cat detection data
def generate_mock_detection():
    """Generate fake cat detection data"""
    # 30% chance of detecting a cat
    if random.random() < 0.3:
        # Generate random box in the frame (assuming 800x600 frame)
        frame_width = 800
        frame_height = 600
        
        # Make the cat appear in different areas of the frame
        x_center = random.randint(100, frame_width - 100)
        y_center = random.randint(100, frame_height - 100)
        
        # Box width and height
        width = random.randint(80, 150)
        height = random.randint(80, 150)
        
        return {
            "detected": True,
            "count": 1,
            "bounding_boxes": [
                {
                    "x": x_center - width // 2,
                    "y": y_center - height // 2,
                    "width": width,
                    "height": height,
                    "confidence": random.uniform(0.6, 0.95),
                    "tracked": False
                }
            ],
            "timestamp": time.time()
        }
    else:
        return {
            "detected": False,
            "count": 0,
            "bounding_boxes": [],
            "timestamp": time.time()
        }

# Generate a mock frame with optional cat detection overlay
def generate_mock_frame(detection=None):
    """Generate a mock camera frame with optional detection overlay"""
    import numpy as np
    import cv2
    
    # Create a dark gray background
    frame = np.ones((600, 800, 3), dtype=np.uint8) * 64
    
    # Add some random noise for texture
    noise = np.random.randint(0, 30, (600, 800, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    # Add a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # If detection data is provided and contains a cat
    if detection and detection.get("detected", False):
        for box in detection.get("bounding_boxes", []):
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            confidence = box.get("confidence", 0.0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Cat: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center crosshair
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
    
    # Add servo position indicator (a small circle in the bottom right)
    pan = mock_state['servo_positions']['pan']
    tilt = mock_state['servo_positions']['tilt']
    
    # Map servo angles to screen coordinates
    screen_x = int(800 * (pan / 180))
    screen_y = int(600 * (tilt / 180))
    
    # Draw servo position indicator
    cv2.circle(frame, (screen_x, screen_y), 5, (0, 140, 255), -1)
    
    # Add laser indicator
    if mock_state['laser_state'] == 'on':
        # Draw a red dot near the servo position
        cv2.circle(frame, (screen_x, screen_y), 3, (0, 0, 255), -1)
        
        # Draw a fuzzy red glow around the laser dot
        for r in range(2, 8, 2):
            cv2.circle(frame, (screen_x, screen_y), r, (0, 0, min(255, 150 + r*20)), 1)
    
    # Convert to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    return img_encoded.tobytes()

# Web routes
@app.route('/')
def index():
    """Main page - redirect to cat tracking UI"""
    return render_template('tracking.html')

@app.route('/cat_tracking')
def cat_tracking_page():
    """Web interface for cat tracking control"""
    return render_template('tracking.html')

@app.route('/cat_tracking/status')
def cat_tracking_status():
    """Return the current status of the cat tracking system"""
    return jsonify({
        'status': 'ok',
        'tracking_active': mock_state['tracking_active'],
        'laser_state': mock_state['laser_state'],
        'pan': mock_state['servo_positions']['pan'],
        'tilt': mock_state['servo_positions']['tilt'],
        'last_detection': mock_state['last_detection'],
        'current_test_mode': mock_state['current_test_mode']
    })

@app.route('/cat_tracking/start')
def cat_tracking_start():
    """Start the cat tracking system"""
    mock_state['tracking_active'] = True
    mock_state['laser_state'] = 'on'
    logger.info("Mock cat tracking started")
    return jsonify({
        'status': 'ok',
        'message': 'Cat tracking system started'
    })

@app.route('/cat_tracking/stop')
def cat_tracking_stop():
    """Stop the cat tracking system"""
    mock_state['tracking_active'] = False
    logger.info("Mock cat tracking stopped")
    return jsonify({
        'status': 'ok',
        'message': 'Cat tracking system stopped'
    })

@app.route('/cat_tracking/center')
def cat_tracking_center():
    """Center the servos"""
    mock_state['servo_positions']['pan'] = 90
    mock_state['servo_positions']['tilt'] = 90
    logger.info("Mock servos centered")
    return jsonify({
        'status': 'ok',
        'message': 'Servos centered'
    })

# Function to run corner test in a thread
def run_corner_test():
    """Mock testing corners in a separate thread"""
    corners = [
        {'pan': 20, 'tilt': 20},  # Top-left
        {'pan': 160, 'tilt': 20},  # Top-right
        {'pan': 160, 'tilt': 160},  # Bottom-right
        {'pan': 20, 'tilt': 160},  # Bottom-left
    ]
    
    mock_state['current_test_mode'] = 'corners'
    
    try:
        for corner in corners:
            if not mock_state['current_test_mode']:
                break  # Test was canceled
                
            logger.info(f"Moving to corner: pan={corner['pan']}, tilt={corner['tilt']}")
            mock_state['servo_positions'] = corner
            time.sleep(1)
            
        # Return to center
        mock_state['servo_positions'] = {'pan': 90, 'tilt': 90}
    finally:
        mock_state['current_test_mode'] = None
        mock_state['test_thread'] = None

@app.route('/cat_tracking/test_corners')
def cat_tracking_test_corners():
    """Test the servo range by moving to the corners"""
    if mock_state['test_thread'] and mock_state['test_thread'].is_alive():
        return jsonify({
            'status': 'error',
            'message': 'A test is already running'
        })
    
    # Start corner test in a thread
    mock_state['test_thread'] = threading.Thread(target=run_corner_test)
    mock_state['test_thread'].daemon = True
    mock_state['test_thread'].start()
    
    return jsonify({
        'status': 'ok',
        'message': 'Corner test started'
    })

@app.route('/remote_control/laser/on')
def remote_control_laser_on():
    """Turn the laser on"""
    mock_state['laser_state'] = 'on'
    logger.info("Mock laser turned ON")
    return jsonify({
        'status': 'ok',
        'laser_state': 'on'
    })

@app.route('/remote_control/laser/off')
def remote_control_laser_off():
    """Turn the laser off"""
    mock_state['laser_state'] = 'off'
    logger.info("Mock laser turned OFF")
    return jsonify({
        'status': 'ok',
        'laser_state': 'off'
    })

@app.route('/detect_cat_json')
def api_detect_cat_json():
    """Mock endpoint for getting cat detection data"""
    detection = generate_mock_detection()
    
    # Save last detection for status endpoint
    if detection["detected"]:
        mock_state['last_detection'] = {
            'timestamp': time.time(),
            'center_x': detection['bounding_boxes'][0]['x'] + detection['bounding_boxes'][0]['width'] // 2,
            'center_y': detection['bounding_boxes'][0]['y'] + detection['bounding_boxes'][0]['height'] // 2,
            'confidence': detection['bounding_boxes'][0]['confidence']
        }
        
        # If tracking is active, move servos toward detection
        if mock_state['tracking_active']:
            # Calculate target servo position
            target_pan = mock_state['servo_positions']['pan']
            target_tilt = mock_state['servo_positions']['tilt']
            
            # Move slightly toward the detection
            if detection['bounding_boxes'][0]['x'] < 400:  # Left side
                target_pan = max(20, mock_state['servo_positions']['pan'] - 10)
            else:  # Right side
                target_pan = min(160, mock_state['servo_positions']['pan'] + 10)
                
            if detection['bounding_boxes'][0]['y'] < 300:  # Top half
                target_tilt = max(20, mock_state['servo_positions']['tilt'] - 10)
            else:  # Bottom half
                target_tilt = min(160, mock_state['servo_positions']['tilt'] + 10)
            
            mock_state['servo_positions'] = {'pan': target_pan, 'tilt': target_tilt}
    
    return jsonify({
        'timestamp': time.time(),
        'detection_result': detection,
        'image_width': 800,
        'image_height': 600
    })

def generate_cat_detection_frames(fps=5):
    """Generator function for mock cat detection stream"""
    frame_delay = 1.0 / fps
    
    while True:
        start_time = time.time()
        
        # Generate mock detection
        detection = generate_mock_detection()
        
        # Generate frame with detection overlay
        frame = generate_mock_frame(detection)
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Sleep to maintain requested FPS
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

@app.route('/stream_cat_detection')
def stream_cat_detection():
    """Endpoint for mock cat detection stream"""
    # Get requested FPS (default to 5)
    fps = float(request.args.get('fps', '5'))
    fps = max(1, min(15, fps))
    
    return Response(
        generate_cat_detection_frames(fps=fps),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stream_metrics')
def stream_metrics():
    """Server-Sent Events endpoint for streaming performance metrics to the browser"""
    def generate_metrics():
        last_metrics = {}
        
        while True:
            # Generate mock detection
            detection = generate_mock_detection()
            
            metrics = {
                'processing_time': random.randint(20, 100),  # Random processing time between 20-100ms
                'cat_detected': detection.get('detected', False),
                'cat_count': detection.get('count', 0),
                'timestamp': int(time.time() * 1000)
            }
            
            # Only send updates when metrics change
            if metrics != last_metrics:
                yield f"data: {json.dumps(metrics)}\n\n"
                last_metrics = metrics.copy()
            
            # Sleep briefly
            time.sleep(0.2)
    
    return Response(generate_metrics(), mimetype="text/event-stream")

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("Starting mock Cat Laser UI Testing Application")
    print("Open http://localhost:5050/ in your web browser")
    
    # Try to install required packages if not available
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("\nMissing required packages. Installing opencv-python...")
        import subprocess
        subprocess.call(["pip", "install", "opencv-python", "numpy"])
        print("Please restart the application now")
        exit(1)
    
    # Start the Flask application on port 5050 to avoid conflict with AirPlay
    app.run(debug=True, port=5050)