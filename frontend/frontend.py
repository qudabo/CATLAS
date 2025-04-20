#!/usr/bin/env python3
"""
Cat Laser Tracking Frontend Server

This script runs a Flask web server that serves the tracking interface and
processes camera images to detect cats using the backend detection service.

Usage:
    python frontend.py [options]

Options:
    --host HOST         Host to bind the server to (default: 0.0.0.0)
    --port PORT         Port to run the server on (default: 5000)
    --camera-index IDX  Camera index to use (default: 0)
    --detection-url URL URL of the detection backend (default: http://127.0.0.1:5001)
    --debug             Enable debug logging
"""

import io
import os
import signal
import subprocess 
import sys
import time
import json
import requests
import threading
import logging
import flask
import RPi.GPIO as GPIO
import queue  # Add queue module for cat tracking
from datetime import datetime

# Add project root to Python path to find shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, jsonify, request, send_file, Response
from picamera2 import Picamera2
from shared.cat_detector import detect_cat

# Import shared configuration
from shared.servo_config import (
    CAMERA_CORNERS, 
    STANDARD_CENTER, 
    SERVO_LIMITS, 
    DEFAULT_GPIO_PINS,
    MOVEMENT_SETTINGS,
    CAMERA_SETTINGS,
    LASER_SETTINGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global camera object
camera = None

# Streaming control flags
streaming_active = False
active_stream_type = None  # Can be 'cat_detection' or 'direct_camera'
active_stream_count = 0
stream_lock = threading.Lock()  # Lock for thread-safe stream management

# Backend server configuration
BACKEND_URL = "http://192.168.2.1:5001"  # Updated to your Mac's IP address


def check_camera_in_use():
    """Check if camera is currently in use by another process"""
    try:
        # Check for processes using the camera
        result = subprocess.run(
            ["fuser", "-v", "/dev/video0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        if result.returncode == 0:
            # Camera is in use
            return True
        return False
    except Exception as e:
        print(f"Error checking camera usage: {e}")
        # If we can't check, assume it might be in use
        return False


def release_camera():
    """Attempt to release the camera by killing processes using it"""
    try:
        print("Attempting to release camera from other processes...")
        # Find processes using the camera
        result = subprocess.run(
            ["fuser", "/dev/video0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )

        if result.returncode == 0 and result.stdout.strip():
            # Get PIDs and kill them
            pids = result.stdout.strip().split()
            print(f"Found processes using camera: {pids}")

            for pid in pids:
                pid = pid.strip()
                if pid and pid.isdigit():
                    # Don't kill our own process
                    if int(pid) != os.getpid():
                        print(f"Killing process {pid}")
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass

            # Give processes time to terminate
            time.sleep(1)
            return True
        return False
    except Exception as e:
        print(f"Error releasing camera: {e}")
        return False


def setup_camera(max_retries=3):
    """Initialize and configure the Pi Camera with specific resolution and framerate"""
    global camera
    
    # If camera is already set up, return it
    if camera is not None:
        return camera
    
    retries = 0
    while retries < max_retries:
        try:
            print(f"Initializing camera (try {retries+1}/{max_retries})...")
            
            # Create camera object
            camera = Picamera2()
            
            # Configure camera with desired resolution and framerate
            # Calculate frame durations in microseconds (for 30fps maximum)
            frame_time_us = int(1000000/30)  # Increased to 30 FPS max for direct camera mode
            
            config = camera.create_still_configuration(
                main={"size": (2304, 1296)},  # Reduced resolution for better performance
                controls={"FrameDurationLimits": (frame_time_us, frame_time_us)}
            )
            camera.configure(config)
            
            # Start the camera
            camera.start()
            
            print("Camera initialized successfully")
            return camera
            
        except Exception as e:
            retries += 1
            print(f"Error initializing camera: {e}")
            time.sleep(1)
            
            # Try to release the camera if it's in use
            if retries < max_retries:
                if release_camera():
                    print("Successfully released camera from other processes")
                    
    print("Failed to initialize camera after multiple attempts")
    return None


def capture_image_to_memory():
    """Capture a still image using the camera and return it as an in-memory buffer"""
    global camera

    # Check if camera is initialized
    if camera is None:
        if not setup_camera():
            raise RuntimeError(
                "Camera is not initialized and could not be initialized")

    # print("Capturing image to memory...")

    # Create an in-memory stream
    image_stream = io.BytesIO()

    # Capture the image to the in-memory stream
    camera.capture_file(image_stream, format="jpeg")

    # Seek to the beginning of the stream
    image_stream.seek(0)

    # print("Image captured to memory buffer")
    return image_stream


@app.route("/capture", methods=["GET"])
def api_capture_image():
    """API endpoint to capture and return an image"""
    try:
        # Capture image directly to memory
        image_stream = capture_image_to_memory()
        return send_file(image_stream, mimetype="image/jpeg", as_attachment=False, download_name="camera_image.jpg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def api_status():
    """API endpoint to check if the camera service is running"""
    global camera
    backend_status = "unknown"
    
    try:
        # Check backend status
        response = requests.get(f"{BACKEND_URL}/status", timeout=2)
        if (response.status_code == 200):
            backend_status = "online"
        else:
            backend_status = "offline"
    except:
        backend_status = "offline"
        
    if camera is not None:
        return jsonify({
            "frontend_status": "online", 
            "backend_status": backend_status,
            "message": "Camera service is running"
        })
    else:
        return jsonify({
            "frontend_status": "offline", 
            "backend_status": backend_status,
            "message": "Camera is not initialized"
        }), 503


@app.route("/info", methods=["GET"])
def api_camera_info():
    """API endpoint to get information about the camera configuration"""
    global camera
    if camera is not None:
        try:
            # Get camera properties
            camera_props = camera.camera_properties
            config = camera.camera_configuration

            # Extract relevant information
            info = {
                "sensor_resolution": f"{camera_props.get('PixelArraySize', 'Unknown')}",
                "current_resolution": f"{config.get('main', {}).get('size', 'Unknown')}",
                "camera_model": camera_props.get("Model", "Unknown"),
            }

            return jsonify(info)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"status": "offline", "message": "Camera is not initialized"}), 503


@app.route("/reset", methods=["POST"])
def api_reset_camera():
    """API endpoint to reset the camera if it's having issues"""
    global camera

    try:
        # Close the camera if it's open
        if camera is not None:
            try:
                camera.close()
            except:
                pass
            camera = None

        # Try to release any processes using the camera
        release_camera()

        # Reinitialize the camera
        if setup_camera():
            return jsonify({"status": "success", "message": "Camera reset successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to reset camera"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect_cat', methods=['GET'])
def api_detect_cat():
    """API endpoint that forwards request to backend for cat detection"""
    try:
        # Capture image
        image_stream = capture_image_to_memory()
        
        # Get confidence threshold from query parameter (default to the backend's default if not provided)
        confidence_threshold = request.args.get('confidence')
        
        # Create multipart form data with the image
        files = {'image': ('image.jpg', image_stream, 'image/jpeg')}
        
        # Forward request to backend with parameters if provided
        params = {}
        if confidence_threshold:
            params['confidence'] = confidence_threshold
        
        # Send request to backend
        response = requests.post(f"{BACKEND_URL}/process_image", files=files, params=params)
        
        if response.status_code != 200:
            return jsonify({"error": "Backend error", "details": response.text}), 500
        
        # Return the backend's response
        return Response(
            response.content, 
            mimetype="image/jpeg",
            headers=dict(response.headers)
        )
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to connect to backend: {str(e)}"}), 503
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/detect_cat_json', methods=['GET'])
def api_detect_cat_json():
    """API endpoint to detect cats and return JSON results for servo control"""
    try:
        # Capture image
        image_stream = capture_image_to_memory()
        
        # Get confidence threshold from query parameter
        confidence_threshold = request.args.get('confidence')
        
        # Create multipart form data with the image
        files = {'image': ('image.jpg', image_stream, 'image/jpeg')}
        
        # Forward request to backend with parameters if provided
        params = {}
        if confidence_threshold:
            params['confidence'] = confidence_threshold
        
        # Send request to backend
        response = requests.post(f"{BACKEND_URL}/process_image_json", files=files, params=params)
        
        if response.status_code != 200:
            return jsonify({"error": "Backend error", "details": response.text}), 500
        
        # Parse the JSON response from backend
        detection_result = response.json()
        
        # Return the JSON response with the correct image dimensions
        return jsonify({
            "timestamp": time.time(),
            "detection_result": detection_result,
            "image_width": CAMERA_SETTINGS['width'],
            "image_height": CAMERA_SETTINGS['height']
        })
        
    except requests.RequestException as e:
        return jsonify({"error": f"Failed to connect to backend: {str(e)}"}), 503
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_cat_detection_frames(fps=5):
    """
    Generator function to continuously capture images, send to backend for processing,
    and stream the annotated images back to the client
    """
    global streaming_active, active_stream_type
    
    # Set the active stream type and mark streaming as active
    with stream_lock:
        streaming_active = True
        active_stream_type = 'cat_detection'
    
    # Calculate the delay between frames based on the desired FPS
    frame_delay = 1.0 / fps
    
    try:
        while streaming_active and active_stream_type == 'cat_detection':
            start_time = time.time()
            
            try:
                # Capture image
                image_stream = capture_image_to_memory()
                
                # Create multipart form data with the image
                files = {'image': ('image.jpg', image_stream, 'image/jpeg')}
                
                # Send request to backend with resize parameter for streaming
                params = {'resize_output': 'true'}
                
                # Send request to backend
                response = requests.post(
                    f"{BACKEND_URL}/process_image", 
                    files=files, 
                    params=params,
                    timeout=5  # Timeout after 5 seconds
                )
                
                if response.status_code == 200:
                    # Extract headers from backend response
                    detected = response.headers.get('X-Cat-Detected', 'false')
                    count = response.headers.get('X-Cat-Count', '0')
                    processing_time = response.headers.get('X-Processing-Time', '0')
                    
                    # Store metrics
                    app.config['LAST_PROCESSING_TIME_MS'] = int(processing_time)
                    app.config['LAST_CAT_DETECTED'] = detected.lower() == 'true'
                    app.config['LAST_CAT_COUNT'] = int(count)
                    
                    # Log detection if a cat was found
                    if detected.lower() == 'true':
                        print(f"Cat detected! Count: {count}, Time: {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Return the annotated image as part of the multipart stream
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(response.content)).encode() + b'\r\n\r\n'
                           + response.content + b'\r\n')
                else:
                    # In case of backend error, display error message
                    error_img = create_error_frame(f"Backend error: {response.status_code}")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
                
                # Calculate elapsed time and sleep if needed to maintain desired FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if (sleep_time > 0):
                    time.sleep(sleep_time)
                    
            except requests.RequestException as e:
                print(f"Error connecting to backend: {e}")
                # Generate error frame for connection issues
                error_img = create_error_frame(f"Backend connection error: {str(e)}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
                # Wait a bit before retrying
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in detection stream: {e}")
                # In case of error, wait a moment before trying again
                time.sleep(0.5)
                
                # Return an error frame
                try:
                    error_img = create_error_frame(str(e))
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
                except:
                    pass
                    
    finally:
        # Only reset streaming flags if this is still the active stream type
        with stream_lock:
            if active_stream_type == 'cat_detection':
                streaming_active = False
                active_stream_type = None


def create_error_frame(error_text):
    """Create a simple image with error text to display in the stream"""
    import numpy as np
    import cv2
    
    # Create a black image
    img = np.zeros((300, 600, 3), np.uint8)
    
    # Write error text
    cv2.putText(img, "Error occurred:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Split long error text into multiple lines
    text_y = 150
    for i in range(0, len(error_text), 40):
        line = error_text[i:i+40]
        cv2.putText(img, line, (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        text_y += 30
    
    # Convert to JPEG
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded.tobytes()


@app.route('/stream_cat_detection')
def stream_cat_detection():
    """Endpoint that streams continuous cat detection at specified FPS"""
    global active_stream_type
    
    # Reset any existing streams to avoid conflicts
    with stream_lock:
        # If there's a different stream active, mark it as inactive
        if active_stream_type is not None and active_stream_type != 'cat_detection':
            streaming_active = False
            # Give a small delay to allow other stream to clean up
            time.sleep(0.1)
    
    # Get requested FPS (default to 5)
    fps = float(request.args.get('fps', '14'))
    
    # Clamp FPS to reasonable values (1-15)
    fps = max(1, min(15, fps))
    
    # Return a multipart response (Motion JPEG)
    response = Response(
        generate_cat_detection_frames(fps=fps),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    
    # Add headers to improve streaming performance
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    
    return response

def generate_direct_camera_frames(fps=30):
    """
    Generator function to stream camera frames directly without cat detection processing
    for higher performance remote control
    """
    global streaming_active, active_stream_type
    
    # Set the active stream type and mark streaming as active
    with stream_lock:
        streaming_active = True
        active_stream_type = 'direct_camera'
    
    # Calculate the delay between frames based on the desired FPS
    frame_delay = 1.0 / fps
    
    try:
        while streaming_active and active_stream_type == 'direct_camera':
            start_time = time.time()
            
            try:
                # Capture image directly
                image_stream = capture_image_to_memory()
                image_data = image_stream.getvalue()
                
                # Return the image directly without any processing
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(image_data)).encode() + b'\r\n\r\n'
                       + image_data + b'\r\n')
                
                # Calculate elapsed time and sleep if needed to maintain desired FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if (sleep_time > 0):
                    # Use a shorter sleep time to maximize responsiveness
                    # This will attempt to get closer to the hardware maximum
                    time.sleep(sleep_time * 0.8)  # Sleep for slightly less than calculated time
                    
            except Exception as e:
                print(f"Error in direct camera stream: {e}")
                # In case of error, wait a moment before trying again
                time.sleep(0.5)
                
                # Return an error frame
                try:
                    error_img = create_error_frame(str(e))
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
                except:
                    pass
                    
    finally:
        # Only reset streaming flags if this is still the active stream type
        with stream_lock:
            if active_stream_type == 'direct_camera':
                streaming_active = False
                active_stream_type = None


@app.route('/stream_direct_camera')
def stream_direct_camera():
    """Endpoint that streams direct camera feed without cat detection for higher performance"""
    global active_stream_type
    
    # Reset any existing streams to avoid conflicts
    with stream_lock:
        # If there's a different stream active, mark it as inactive
        if active_stream_type is not None and active_stream_type != 'direct_camera':
            streaming_active = False
            # Give a small delay to allow other stream to clean up
            time.sleep(0.1)
    
    # Use a fixed high FPS value for better remote control responsiveness
    fps = 30.0  # Increased from 20 to 30 FPS for maximum performance
    
    # Return a multipart response (Motion JPEG)
    response = Response(
        generate_direct_camera_frames(fps=fps),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    
    # Add headers to improve streaming performance
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    
    return response

# local site
@app.route('/cat_detection_viewer')
def cat_detection_viewer():
    """Endpoint that shows an optimized HTML page to view the cat detection stream in real-time with minimal latency"""
    # Pass FPS value to the template
    fps = request.args.get('fps', '5')
    
    return flask.render_template('cat_detection_viewer.html', fps=fps)

@app.route('/stream_metrics')
def stream_metrics():
    """
    Server-Sent Events endpoint that streams performance metrics to the browser.
    This allows the browser to display processing time and detection info without
    the overhead of parsing image headers.
    """
    def generate_metrics():
        last_metrics = {}
        
        try:
            while True:
                # Get latest metrics from application state
                metrics = {
                    'processing_time': app.config.get('LAST_PROCESSING_TIME_MS', 0),
                    'cat_detected': app.config.get('LAST_CAT_DETECTED', False),
                    'cat_count': app.config.get('LAST_CAT_COUNT', 0),
                    'timestamp': int(time.time() * 1000)
                }
                
                # Only send updates when metrics change
                if metrics != last_metrics:
                    yield f"data: {json.dumps(metrics)}\n\n"
                    last_metrics = metrics.copy()
                
                # Sleep briefly to reduce CPU usage
                time.sleep(0.1)
                
        except GeneratorExit:
            pass
    
    return Response(generate_metrics(), mimetype="text/event-stream")


@app.route('/remote_control')
def remote_control():
    """Web interface for remote control of the cat laser"""
    return flask.render_template('remote_control.html')

# Remote control API endpoints
@app.route('/remote_control/status')
def remote_control_status():
    """Return current status of the servo motors"""
    global camera
    
    # Import and initialize servo controller if needed
    from servo_controller import DualServoController
    
    # Use singleton pattern to create servo controller only once
    if not hasattr(app, 'servo_controller'):
        try:
            # Default GPIO pins to match existing code
            app.servo_controller = DualServoController(pan_pin=18, tilt_pin=17)
            app.servo_controller.center()  # Center on startup
        except Exception as e:
            logger.error(f"Error initializing servo controller: {e}")
            return jsonify({'status': 'error', 'message': f'Error initializing servo controller: {e}'})
    
    # Initialize laser controller if needed
    initialize_controllers()
    
    # Get laser state
    laser_state = "on" if app.laser_controller.is_on else "off"
    
    return jsonify({
        'status': 'ok',
        'current_pan': app.servo_controller.current_pan,
        'current_tilt': app.servo_controller.current_tilt,
        'pan_limits': {'min': app.servo_controller.pan_min, 'max': app.servo_controller.pan_max},
        'tilt_limits': {'min': app.servo_controller.tilt_min, 'max': app.servo_controller.tilt_max},
        'laser_state': laser_state
    })

def initialize_servo_controller():
    """Initialize the servo controller if not already initialized"""
    if not hasattr(app, 'servo_controller'):
        try:
            from servo_controller import DualServoController
            
            # Initialize the servo controller with proper GPIO setup
            logger.info(f"Initializing servo controller with pan_pin={DEFAULT_GPIO_PINS['pan_pin']}, tilt_pin={DEFAULT_GPIO_PINS['tilt_pin']}")
            app.servo_controller = DualServoController(
                pan_pin=DEFAULT_GPIO_PINS['pan_pin'],
                tilt_pin=DEFAULT_GPIO_PINS['tilt_pin']
            )
            
            # Apply movement settings
            app.servo_controller.pan_step = MOVEMENT_SETTINGS.get('pan_step', 1.0)
            app.servo_controller.tilt_step = MOVEMENT_SETTINGS.get('tilt_step', 1.0)
            app.servo_controller.movement_delay = MOVEMENT_SETTINGS.get('movement_delay', 0.01)
            
            # Mark as initialized 
            app.servo_controller_initialized = True
            
            logger.info("Servo controller initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing servo controller: {e}")
            return False
    
    return True

@app.route('/remote_control/move', methods=['POST'])
def remote_control_move():
    """Move the servo to the specified position"""
    # Initialize servo controller if needed
    if not initialize_servo_controller():
        return jsonify({'status': 'error', 'message': 'Failed to initialize servo controller'})
    
    data = flask.request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'})
    
    # Check if we have normalized coordinates (0-1)
    if 'x' in data and 'y' in data:
        x_norm = max(0, min(1, data['x']))
        y_norm = max(0, min(1, data['y']))
        
        # Bilinear interpolation for pan angle
        pan_top = CAMERA_CORNERS['top_left']['pan'] + x_norm * (CAMERA_CORNERS['top_right']['pan'] - CAMERA_CORNERS['top_left']['pan'])
        pan_bottom = CAMERA_CORNERS['bottom_left']['pan'] + x_norm * (CAMERA_CORNERS['bottom_right']['pan'] - CAMERA_CORNERS['bottom_left']['pan'])
        pan_angle = pan_top + y_norm * (pan_bottom - pan_top)
        
        # Bilinear interpolation for tilt angle
        tilt_left = CAMERA_CORNERS['top_left']['tilt'] + y_norm * (CAMERA_CORNERS['bottom_left']['tilt'] - CAMERA_CORNERS['top_left']['tilt'])
        tilt_right = CAMERA_CORNERS['top_right']['tilt'] + y_norm * (CAMERA_CORNERS['bottom_right']['tilt'] - CAMERA_CORNERS['top_right']['tilt'])
        tilt_angle = tilt_left + x_norm * (tilt_right - tilt_left)
        
    elif 'pan' in data and 'tilt' in data:
        # Direct servo angles
        pan_angle = data['pan']
        tilt_angle = data['tilt']
    else:
        return jsonify({'status': 'error', 'message': 'Invalid data format'})
    
    # Ensure angles are within valid range
    pan_angle = max(app.servo_controller.pan_min, min(app.servo_controller.pan_max, pan_angle))
    tilt_angle = max(app.servo_controller.tilt_min, min(app.servo_controller.tilt_max, tilt_angle))
    
    logger.info(f"Moving to pan={pan_angle:.1f}, tilt={tilt_angle:.1f}")
    
    # Move servos directly without smoothing
    try:
        app.servo_controller.move(pan_angle, tilt_angle)
        return jsonify({
            'status': 'ok',
            'pan': pan_angle,
            'tilt': tilt_angle
        })
    except Exception as e:
        logger.error(f"Error moving servo: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/remote_control/center')
def remote_control_center():
    """Center the servos"""
    # Import and initialize servo controller if needed
    from servo_controller import DualServoController
    
    # Use singleton pattern to create servo controller only once
    if not hasattr(app, 'servo_controller'):
        try:
            # Use GPIO pins from configuration
            app.servo_controller = DualServoController(
                pan_pin=DEFAULT_GPIO_PINS['pan_pin'],
                tilt_pin=DEFAULT_GPIO_PINS['tilt_pin']
            )
        except Exception as e:
            logger.error(f"Error initializing servo controller: {e}")
            return jsonify({'status': 'error', 'message': f'Error initializing servo controller: {e}'})
    
    logger.info("Centering servos")
    app.servo_controller.center()
    return jsonify({
        'status': 'ok',
        'pan': STANDARD_CENTER['pan'],
        'tilt': STANDARD_CENTER['tilt']
    })

@app.route('/remote_control/corners')
def remote_control_corners():
    """Test by moving to all corners"""
    # Import and initialize servo controller if needed
    from servo_controller import DualServoController
    
    # Use singleton pattern to create servo controller only once
    if not hasattr(app, 'servo_controller'):
        try:
            # Use GPIO pins from configuration
            app.servo_controller = DualServoController(
                pan_pin=DEFAULT_GPIO_PINS['pan_pin'],
                tilt_pin=DEFAULT_GPIO_PINS['tilt_pin']
            )
        except Exception as e:
            logger.error(f"Error initializing servo controller: {e}")
            return jsonify({'status': 'error', 'message': f'Error initializing servo controller: {e}'})
    
    # Start a thread to move through corners
    def corner_sequence():
        logger.info("Moving to calibrated camera corners")
        
        # Define the order of corners to visit
        corners = [
            ('top_left', CAMERA_CORNERS['top_left']),
            ('top_right', CAMERA_CORNERS['top_right']),
            ('bottom_right', CAMERA_CORNERS['bottom_right']),
            ('bottom_left', CAMERA_CORNERS['bottom_left']),
        ]
        
        # Move to each corner with a delay
        for name, position in corners:
            logger.info(f"Moving to {name}: pan={position['pan']}, tilt={position['tilt']}")
            app.servo_controller.move(position['pan'], position['tilt'])
            time.sleep(1.0)
        
        # Return to center
        logger.info("Returning to center")
        app.servo_controller.center()
    
    threading.Thread(target=corner_sequence, daemon=True).start()
    return jsonify({'status': 'ok'})


class LaserController:
    """Class to control the laser module with 2N3906 PNP transistor"""
    
    def __init__(self, pin=DEFAULT_GPIO_PINS['laser_pin']):
        """Initialize the laser controller"""
        self.pin = pin
        self.is_on = False
        self.last_activity_time = time.time()
        
        # Setup GPIO with BCM mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)  # Disable warnings that might confuse users
        
        # Setup laser pin as output (HIGH = laser OFF, LOW = laser ON)
        # For PNP transistor, HIGH turns laser OFF (initial state)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH)
        logger.info(f"Laser controller initialized with pin {self.pin}")
        
        # Start the auto-off monitoring thread
    
    def turn_on(self):
        """Turn the laser on"""
        # For PNP transistor: Drive pin LOW to turn laser ON
        GPIO.output(self.pin, GPIO.LOW)
        self.is_on = True
        self.last_activity_time = time.time()
        logger.info("Laser turned ON")
        return True
    
    def turn_off(self):
        """Turn the laser off"""
        # For PNP transistor: Drive pin HIGH to turn laser OFF
        GPIO.output(self.pin, GPIO.HIGH)
        self.is_on = False
        logger.info("Laser turned OFF")
        return True
    
    def toggle(self):
        """Toggle the laser state"""
        if self.is_on:
            return self.turn_off()
        else:
            return self.turn_on()

    def register_activity(self):
        """Register user activity to prevent auto-turnoff"""
        self.last_activity_time = time.time()
        
    def cleanup(self):
        """Clean up resources"""
        # Ensure laser is off before cleanup
        if self.is_on:
            self.turn_off()

# Initialize the laser controller during app startup
def initialize_controllers():
    """Initialize both servo and laser controllers if needed"""
    if not initialize_servo_controller():
        return False
    
    if not hasattr(app, 'laser_controller'):
        try:
            # Initialize the laser controller
            logger.info(f"Initializing laser controller with pin {DEFAULT_GPIO_PINS['laser_pin']}")
            app.laser_controller = LaserController(
                pin=DEFAULT_GPIO_PINS['laser_pin'],
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing laser controller: {e}")
            return False
    
    return True

@app.route('/remote_control/laser/on')
def remote_control_laser_on():
    """Turn the laser on"""
    # Initialize controllers if needed
    if not initialize_controllers():
        return jsonify({'status': 'error', 'message': 'Failed to initialize laser controller'})
    
    try:
        app.laser_controller.turn_on()
        return jsonify({'status': 'ok', 'laser_state': 'on'})
    except Exception as e:
        logger.error(f"Error turning laser on: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/remote_control/laser/off')
def remote_control_laser_off():
    """Turn the laser off"""
    # Initialize controllers if needed
    if not initialize_controllers():
        return jsonify({'status': 'error', 'message': 'Failed to initialize laser controller'})
    
    try:
        app.laser_controller.turn_off()
        return jsonify({'status': 'ok', 'laser_state': 'off'})
    except Exception as e:
        logger.error(f"Error turning laser off: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/remote_control/laser/toggle')
def remote_control_laser_toggle():
    """Toggle the laser state"""
    # Initialize controllers if needed
    if not initialize_controllers():
        return jsonify({'status': 'error', 'message': 'Failed to initialize laser controller'})
    
    try:
        app.laser_controller.toggle()
        laser_state = 'on' if app.laser_controller.is_on else 'off'
        return jsonify({'status': 'ok', 'laser_state': laser_state})
    except Exception as e:
        logger.error(f"Error toggling laser: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


class CatLaserTracker:
    """Main controller for the cat laser tracking system"""
    
    def __init__(self, backend_url, frontend_url, pan_pin, tilt_pin, poll_rate=0.2, 
                 image_width=CAMERA_SETTINGS['width'], image_height=CAMERA_SETTINGS['height'], 
                 pan_speed=MOVEMENT_SETTINGS['pan_step'], tilt_speed=MOVEMENT_SETTINGS['tilt_step'], 
                 move_delay=MOVEMENT_SETTINGS['movement_delay']):
        """Initialize the laser tracking system"""
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.poll_rate = poll_rate
        
        # Image dimensions - updated for high resolution camera
        self.image_width = image_width
        self.image_height = image_height
        
        # Save speed parameters for easier access later
        self.pan_speed = pan_speed
        self.tilt_speed = tilt_speed
        self.move_delay = move_delay
        
        # Create servo controller
        logger.info(f"Initializing servo controller with pan pin {pan_pin}, tilt pin {tilt_pin}")
        logger.info(f"Servo speed settings - Pan: {pan_speed} deg/move, Tilt: {tilt_speed} deg/move, Delay: {move_delay}s")
        
        # Use app.servo_controller if it exists, otherwise create a new one
        from servo_controller import DualServoController
        # If we're using app instance, use servo_controller from there
        if hasattr(app, 'servo_controller'):
            self.servo = app.servo_controller
            logger.info("Using existing servo controller from Flask app")
        else:
            self.servo = DualServoController(pan_pin=pan_pin, tilt_pin=tilt_pin)
        
        # Set servo movement parameters
        self.servo.pan_step = pan_speed
        self.servo.tilt_step = tilt_speed
        self.servo.movement_delay = move_delay
        
        # Initialize laser controller
        if hasattr(app, 'laser_controller'):
            self.laser = app.laser_controller
            logger.info("Using existing laser controller from Flask app")
        else:
            logger.info("Initializing laser controller")
            self.laser = LaserController(pin=DEFAULT_GPIO_PINS['laser_pin'])
            
        # Turn on laser at startup
        self.laser.turn_on()
        
        self.tracking_active = False
        self.tracking_thread = None
        self.running = False
        
        # Custom coordinate mapping with calibrated corners
        self.map_coordinates_to_angles = self.calibrated_map_coordinates_to_angles
        
        # Calculate camera center point from the corners
        self.camera_center = {
            'x': self.image_width / 2,
            'y': self.image_height / 2
        }
        
        # Queue for tracking detected cats
        self.detection_queue = queue.Queue(maxsize=10)
        # Initialize last detection time
        self.last_detection_time = 0
        
        # Test connections
        self._test_connections()
    
    def calibrated_map_coordinates_to_angles(self, x, y):
        """
        Map image coordinates (x, y) to servo angles using bilinear interpolation 
        from the calibrated corner positions.
        
        Args:
            x: X coordinate in the image (0 to image_width)
            y: Y coordinate in the image (0 to image_height)
            
        Returns:
            tuple: (pan_angle, tilt_angle) interpolated from the calibrated corners
        """
        # Normalize coordinates to [0, 1] range
        x_norm = max(0, min(1, x / self.image_width))
        y_norm = max(0, min(1, y / self.image_height))
        
        # Define our corners with their calibrated angles
        # The corners are arranged in this order: [top_left, top_right, bottom_left, bottom_right]
        corners = {
            'top_left': {'x': 0, 'y': 0, 'pan': CAMERA_CORNERS['top_left']['pan'], 'tilt': CAMERA_CORNERS['top_left']['tilt']},
            'top_right': {'x': self.image_width, 'y': 0, 'pan': CAMERA_CORNERS['top_right']['pan'], 'tilt': CAMERA_CORNERS['top_right']['tilt']},
            'bottom_left': {'x': 0, 'y': self.image_height, 'pan': CAMERA_CORNERS['bottom_left']['pan'], 'tilt': CAMERA_CORNERS['bottom_left']['tilt']},
            'bottom_right': {'x': self.image_width, 'y': self.image_height, 'pan': CAMERA_CORNERS['bottom_right']['pan'], 'tilt': CAMERA_CORNERS['bottom_right']['tilt']}
        }
        
        # Bilinear interpolation for pan angle
        # First, interpolate along the top edge (between top_left and top_right)
        pan_top = corners['top_left']['pan'] + x_norm * (corners['top_right']['pan'] - corners['top_left']['pan'])
        
        # Then, interpolate along the bottom edge (between bottom_left and bottom_right)
        pan_bottom = corners['bottom_left']['pan'] + x_norm * (corners['bottom_right']['pan'] - corners['bottom_left']['pan'])
        
        # Finally, interpolate between those two values based on y coordinate
        pan_angle = pan_top + y_norm * (pan_bottom - pan_top)
        
        # Bilinear interpolation for tilt angle
        # First, interpolate along the left edge (between top_left and bottom_left)
        tilt_left = corners['top_left']['tilt'] + y_norm * (corners['bottom_left']['tilt'] - corners['top_left']['tilt'])
        
        # Then, interpolate along the right edge (between top_right and bottom_right)
        tilt_right = corners['top_right']['tilt'] + y_norm * (corners['bottom_right']['tilt'] - corners['top_right']['tilt'])
        
        # Finally, interpolate between those two values based on x coordinate
        tilt_angle = tilt_left + x_norm * (tilt_right - tilt_left)
        
        # Log the mapping values for debugging
        logger.debug(f"Cat at pixel ({x}, {y}) -> normalized ({x_norm:.2f}, {y_norm:.2f})")
        logger.debug(f"Mapped to servo angles (pan={pan_angle:.1f}, tilt={tilt_angle:.1f})")
        
        return (pan_angle, tilt_angle)
        
    def _test_connections(self):
        """Test connections to frontend and backend before starting full operation"""
        logger.info(f"Testing connection to frontend: {self.frontend_url}")
        try:
            response = requests.get(f"{self.frontend_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info("Frontend connection successful")
            else:
                logger.warning(f"Frontend responded with status code: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"WARNING: Cannot connect to frontend: {e}")
            logger.error(f"Make sure frontend server is running at {self.frontend_url}")
            logger.error("The system may not work correctly without a frontend connection")
        
        logger.info(f"Testing connection to backend: {self.backend_url}")
        try:
            response = requests.get(f"{self.backend_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info("Backend connection successful")
            else:
                logger.warning(f"Backend responded with status code: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"WARNING: Cannot connect to backend: {e}")
            logger.error(f"Make sure backend server is running at {self.backend_url}")
            logger.error("The system may not work correctly without a backend connection")
        
    def start(self):
        """Start the cat laser tracking system"""
        if self.tracking_thread is not None and self.tracking_thread.is_alive():
            logger.warning("Cat laser tracking is already running")
            return
            
        logger.info("Starting cat laser tracking system")
        self.running = True
        
        # Start the tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        # Start servo tracking thread that consumes the queue
        self.servo.start_tracking(detection_queue=self.detection_queue, poll_interval=self.poll_rate)
        
        logger.info("Cat laser tracking system started")
        
    def stop(self):
        """Stop the cat laser tracking system"""
        if not self.running:
            return
            
        logger.info("Stopping cat laser tracking system")
        self.running = False
        
        # Stop servo tracking
        self.servo.stop_tracking()
        
        # Wait for tracking thread to finish
        if self.tracking_thread is not None and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2.0)
            
        logger.info("Cat laser tracking system stopped")
        
    def _tracking_loop(self):
        """Main tracking loop that polls for cat detections"""
        logger.info("Starting tracking loop")
        consecutive_errors = 0
        max_retry_delay = 10  # Maximum seconds to wait between retries
        
        while self.running:
            try:
                # Get the latest cat detection from the frontend
                logger.debug(f"Requesting cat detection data from {self.frontend_url}/detect_cat_json")
                response = requests.get(
                    f"{self.frontend_url}/detect_cat_json",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    # Reset error counter on successful request
                    consecutive_errors = 0
                    
                    data = response.json()
                    detection_result = data.get("detection_result", {})
                    
                    # Update image dimensions if available
                    if "image_width" in data and "image_height" in data:
                        if data["image_width"] != self.image_width or data["image_height"] != self.image_height:
                            new_width = data["image_width"]
                            new_height = data["image_height"]
                            logger.info(f"Updating image dimensions: {new_width}x{new_height}")
                            self.image_width = new_width
                            self.image_height = new_height
                            self.servo.update_image_dimensions(new_width, new_height)
                    
                    # Process detection result
                    if detection_result.get("detected", False):
                        # Get the bounding boxes
                        bounding_boxes = detection_result.get("bounding_boxes", [])
                        if bounding_boxes:
                            # Get the first (highest confidence) box
                            box = bounding_boxes[0]
                            
                            # Calculate point below the bottom middle of bounding box
                            bottom_middle_x = box["x"] + (box["width"] // 2)
                            
                            # Position the laser 20% of the box height below the bottom of the box
                            # This gives an extra safety margin beyond just the bottom middle
                            safety_offset = int(box["height"] * 0.2)  # 20% of box height as extra buffer
                            bottom_middle_y = box["y"] + box["height"] + safety_offset
                            
                            # Log detection with adjusted aim point
                            logger.info(f"Cat detected! Aiming lower for safety ({bottom_middle_x}, {bottom_middle_y}), confidence: {box.get('confidence', 0):.2f}")
                            
                            # Use our calibrated mapping function to convert coordinates to angles
                            pan_angle, tilt_angle = self.calibrated_map_coordinates_to_angles(bottom_middle_x, bottom_middle_y)
                            logger.info(f"Mapped to servo angles: pan={pan_angle:.1f}, tilt={tilt_angle:.1f}")
                            
                            # Create a modified box with pre-calculated angles
                            box_with_angles = box.copy()
                            box_with_angles['pan_angle'] = pan_angle
                            box_with_angles['tilt_angle'] = tilt_angle
                            
                            # Queue this detection for servo control
                            try:
                                if not self.detection_queue.full():
                                    self.detection_queue.put_nowait(box_with_angles)
                                    self.last_detection_time = time.time()
                            except queue.Full:
                                # If queue is full, drop this detection
                                pass
                    else:
                        # No cat detected in this frame
                        time_since_detection = time.time() - self.last_detection_time
                        if time_since_detection > 5.0 and self.last_detection_time > 0:
                            logger.debug(f"No cats detected for {time_since_detection:.1f} seconds")
                            
                else:
                    logger.warning(f"Unexpected response from frontend: {response.status_code}")
                    consecutive_errors += 1
                
            except requests.RequestException as e:
                consecutive_errors += 1
                retry_delay = min(self.poll_rate * 2**consecutive_errors, max_retry_delay)
                
                logger.error(f"Error connecting to frontend: {e}")
                logger.info(f"Retrying in {retry_delay:.1f} seconds (attempt {consecutive_errors})")
                
                time.sleep(retry_delay)
                continue  # Skip the normal poll_rate delay
                
            except Exception as e:
                logger.exception(f"Error in tracking loop: {e}")
                consecutive_errors += 1
            
            # Wait before next detection
            time.sleep(self.poll_rate)
            
        logger.info("Tracking loop stopped")
    
    def center(self):
        """Center the laser to the standard center position (90,90)"""
        logger.info(f"Centering to standard position: pan={STANDARD_CENTER['pan']}, tilt={STANDARD_CENTER['tilt']}")
        self.servo.move(STANDARD_CENTER['pan'], STANDARD_CENTER['tilt'])
        
    def test_corners(self):
        """Test the servo range by moving to the calibrated corners"""
        logger.info("Moving to calibrated camera corners")
        
        # Define the order of corners to visit
        corners = [
            ('top_left', CAMERA_CORNERS['top_left']),
            ('top_right', CAMERA_CORNERS['top_right']),
            ('bottom_right', CAMERA_CORNERS['bottom_right']),
            ('bottom_left', CAMERA_CORNERS['bottom_left']),
        ]
        
        # Move to each corner with a delay
        for name, position in corners:
            logger.info(f"Moving to {name}: pan={position['pan']}, tilt={position['tilt']}")
            self.servo.move(position['pan'], position['tilt'])
            time.sleep(1.0)
        
        # Return to calibrated center
        logger.info("Returning to center")
        self.center()
        
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        self.stop()
        
        # Turn off the laser
        if hasattr(self, 'laser'):
            logger.info("Turning off laser")
            self.laser.turn_off()


# Flask app singleton for the cat laser tracking system
def get_cat_laser_tracker():
    """Get or initialize the cat laser tracker singleton"""
    if not hasattr(app, 'cat_laser_tracker'):
        try:
            # Initialize the controller
            logger.info("Initializing cat laser tracker")
            initialize_controllers()  # Make sure controllers are initialized
            
            # Create the tracker with default settings
            app.cat_laser_tracker = CatLaserTracker(
                backend_url=BACKEND_URL,
                frontend_url="http://127.0.0.1:5000",  # Local frontend URL since we're in the same app
                pan_pin=DEFAULT_GPIO_PINS['pan_pin'],
                tilt_pin=DEFAULT_GPIO_PINS['tilt_pin'],
                poll_rate=0.2,
                image_width=CAMERA_SETTINGS['width'],
                image_height=CAMERA_SETTINGS['height'],
                pan_speed=MOVEMENT_SETTINGS['pan_step'],
                tilt_speed=MOVEMENT_SETTINGS['tilt_step'],
                move_delay=MOVEMENT_SETTINGS['movement_delay']
            )
            logger.info("Cat laser tracker initialized")
        except Exception as e:
            logger.error(f"Error initializing cat laser tracker: {e}")
            return None
    
    return app.cat_laser_tracker

@app.route('/cat_tracking')
def cat_tracking_page():
    """Web interface for cat tracking control"""
    return flask.render_template('cat_tracking.html')

@app.route('/cat_tracking/status')
def cat_tracking_status():
    """Return the current status of the cat tracking system"""
    tracker = get_cat_laser_tracker()
    if not tracker:
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize cat tracking system'
        }), 500
    
    # Get laser state
    initialize_controllers()
    laser_state = "on" if app.laser_controller.is_on else "off"
    
    return jsonify({
        'status': 'ok',
        'tracking_active': hasattr(tracker, 'running') and tracker.running,
        'laser_state': laser_state
    })

@app.route('/cat_tracking/start')
def cat_tracking_start():
    """Start the cat tracking system"""
    tracker = get_cat_laser_tracker()
    if not tracker:
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize cat tracking system'
        }), 500
    
    try:
        # Make sure laser is on
        tracker.laser.turn_on()
        # Start tracking
        tracker.start()
        return jsonify({
            'status': 'ok',
            'message': 'Cat tracking system started'
        })
    except Exception as e:
        logger.error(f"Error starting cat tracking: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/cat_tracking/stop')
def cat_tracking_stop():
    """Stop the cat tracking system"""
    tracker = get_cat_laser_tracker()
    if not tracker:
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize cat tracking system'
        }), 500
    
    try:
        tracker.stop()
        return jsonify({
            'status': 'ok',
            'message': 'Cat tracking system stopped'
        })
    except Exception as e:
        logger.error(f"Error stopping cat tracking: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/cat_tracking/center')
def cat_tracking_center():
    """Center the servos"""
    tracker = get_cat_laser_tracker()
    if not tracker:
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize cat tracking system'
        }), 500
    
    try:
        tracker.center()
        return jsonify({
            'status': 'ok',
            'message': 'Servos centered'
        })
    except Exception as e:
        logger.error(f"Error centering servos: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/cat_tracking/test_corners')
def cat_tracking_test_corners():
    """Test the servo range by moving to the calibrated corners"""
    tracker = get_cat_laser_tracker()
    if not tracker:
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize cat tracking system'
        }), 500
    
    try:
        # Run in a separate thread to avoid blocking
        threading.Thread(
            target=tracker.test_corners,
            daemon=True
        ).start()
        
        return jsonify({
            'status': 'ok',
            'message': 'Testing servo corners'
        })
    except Exception as e:
        logger.error(f"Error testing corners: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/CATLAS')
def catlas_interface():
    """Unified interface for the CATLAS (Cat Automated Tracking Laser System)
    Combines all functionality into a single modern web interface"""
    return flask.render_template('catlas.html')


def cleanup():
    """Clean up resources before exiting"""
    global camera
    if camera is not None:
        try:
            print("Stopping camera...")
            camera.close()
        except Exception as e:
            print(f"Error stopping camera: {e}")


def main():
    print("Initializing frontend camera system...")
    print(f"Will connect to backend at: {BACKEND_URL}")

    # Register cleanup function to run on exit
    import atexit
    atexit.register(cleanup)

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("Received shutdown signal, cleaning up...")
        # Turn off laser when terminating
        try:
            if hasattr(app, 'laser_controller'):
                print("Turning off laser as part of shutdown...")
                app.laser_controller.turn_off()
        except Exception as e:
            print(f"Error turning off laser during shutdown: {e}")
            
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set up camera
    if setup_camera() is None:
        print("Warning: Camera initialization failed. Starting server anyway.")
        print("You can try to initialize the camera later using the /reset endpoint.")
    
    # Initialize controllers and turn on laser at startup
    print("Initializing laser controller and turning on laser...")
    initialize_controllers()
    app.laser_controller.turn_on()
    print("Laser activated!")

    # Run the Flask app - using port 5000 for frontend
    app.run(host="0.0.0.0", port=5000)  


if __name__ == "__main__":
    main()
