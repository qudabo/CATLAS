import io
import os
import signal
import subprocess 
import sys
import time
import threading
import json
from datetime import datetime

from flask import Flask, jsonify, request, send_file, Response
from picamera2 import Picamera2

# Import the cat detector module
from cat_detector import detect_cat

app = Flask(__name__)

# Global camera object
camera = None

# Streaming control flag
streaming_active = False


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
    """Initialize and configure the Pi Camera Module 3 with retry mechanism"""
    global camera

    # If camera is already set up, return it
    if camera is not None:
        return camera

    retries = 0
    while retries < max_retries:
        try:
            print(
                f"Attempting to initialize camera (try {retries+1}/{max_retries})...")

            # Check if camera is in use and try to release it if needed
            if check_camera_in_use():
                print("Camera appears to be in use by another process")
                if not release_camera():
                    print("Failed to release camera, will retry...")
                    retries += 1
                    time.sleep(2)
                    continue

            # Create camera object
            camera = Picamera2()

            # Get camera properties and capabilities
            camera_props = camera.camera_properties
            print(
                f"Sensor max resolution: {camera_props.get('PixelArraySize', 'Unknown')}")

            # Configure the camera with still capture configuration for high resolution
            still_config = camera.create_still_configuration()

            # Make sure we're using the full sensor resolution
            # Get sensor dimensions if available
            if "PixelArraySize" in camera_props:
                width, height = camera_props["PixelArraySize"]
                # Ensure width and height are multiples of 2 (required by some encoders)
                width = width - (width % 2)
                height = height - (height % 2)
                # Update the still configuration with full resolution
                still_config["main"]["size"] = (width, height)
                print(f"Setting camera to full resolution: {width}x{height}")

            # Configure camera with still configuration
            camera.configure(still_config)

            # Set autofocus mode to continuous (2 = continuous)
            camera.set_controls({"AfMode": 2})

            # Start the camera and keep it running
            camera.start()

            # Allow time for auto exposure and focus
            time.sleep(2)
            print("Camera initialized and ready")
            print(f"Camera resolution: {still_config['main']['size']}")

            return camera

        except Exception as e:
            print(f"Camera initialization failed: {e}")
            retries += 1

            # Try to release camera on next attempt
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)

    # If we reach here, we failed to initialize the camera
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

    print("Capturing image to memory...")

    # Create an in-memory stream
    image_stream = io.BytesIO()

    # Capture the image to the in-memory stream
    # Note: quality parameter is not supported by Picamera2.capture_file
    camera.capture_file(image_stream, format="jpeg")

    # Seek to the beginning of the stream
    image_stream.seek(0)

    print("Image captured to memory buffer")
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
    if camera is not None:
        return jsonify({"status": "online", "message": "Camera service is running"})
    else:
        return jsonify({"status": "offline", "message": "Camera is not initialized"}), 503


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


@app.route("/detect_cat", methods=["GET"])
def api_detect_cat():
    """API endpoint to capture an image, detect cats, and return the annotated image"""
    try:
        # Capture image
        image_stream = capture_image_to_memory()
        
        # Get confidence threshold from query parameter (default to 0.5 if not provided)
        confidence_threshold = float(request.args.get('confidence', '0.5'))
        
        # Run cat detection with specified confidence threshold
        annotated_image, detection_result = detect_cat(image_stream, confidence_threshold)
        
        # Create a new BytesIO object with the annotated image
        annotated_stream = io.BytesIO(annotated_image)
        
        # Log detection results
        print(f"Cat detection result: {detection_result}")
        
        # Return image with detection results in headers
        response = send_file(
            annotated_stream, 
            mimetype="image/jpeg", 
            as_attachment=False, 
            download_name="cat_detection.jpg"
        )
        
        # Add detection results as headers
        response.headers['X-Cat-Detected'] = str(detection_result['detected'])
        response.headers['X-Cat-Count'] = str(detection_result['count'])
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/detect_cat_json", methods=["GET"])
def api_detect_cat_json():
    """API endpoint to capture an image, detect cats, and return detailed JSON results"""
    try:
        # Capture image
        image_stream = capture_image_to_memory()
        
        # Run cat detection
        _, detection_result = detect_cat(image_stream)
        
        # Return detection results as JSON
        return jsonify({
            "detection_result": detection_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_cat_detection_frames(fps=5):
    """
    Generator function to continuously capture images and detect cats
    at the specified frames per second rate with optimizations for reduced latency.
    """
    global streaming_active
    streaming_active = True
    
    # Calculate the delay between frames based on the desired FPS
    frame_delay = 1.0 / fps
    
    try:
        while streaming_active:
            start_time = time.time()
            
            try:
                # Capture image
                image_stream = capture_image_to_memory()
                
                # Get confidence threshold from app config or use default
                confidence_threshold = app.config.get('CAT_CONFIDENCE_THRESHOLD', 0.5)
                
                # Run cat detection with confidence threshold and resize for browser
                annotated_image, detection_result = detect_cat(
                    image_stream, 
                    confidence_threshold,
                    resize_output=True  # Resize to browser-friendly dimensions
                )
                
                # Store metrics for SSE endpoint
                app.config['LAST_PROCESSING_TIME_MS'] = detection_result.get('processing_time_ms', 0)
                app.config['LAST_CAT_DETECTED'] = detection_result['detected']
                app.config['LAST_CAT_COUNT'] = detection_result['count']
                
                # Log detection if a cat was found
                if detection_result['detected']:
                    print(f"Cat detected! Count: {detection_result['count']}, Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Add headers to enable better browser caching
                headers = {
                    'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'X-Cat-Detected': str(detection_result['detected']),
                    'X-Cat-Count': str(detection_result['count']),
                    'X-Processing-Time': str(detection_result.get('processing_time_ms', 0))
                }
                
                # Return the annotated image as part of the multipart stream
                # Smaller content-type header to reduce overhead
                yield (b'--frame\r\n'
                       b'Content-Type:image/jpeg\r\n'
                       b'Content-Length: ' + str(len(annotated_image)).encode() + b'\r\n\r\n'
                       + annotated_image + b'\r\n')
                
                # Calculate elapsed time and sleep if needed to maintain desired FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if (sleep_time > 0):
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error in detection stream: {e}")
                # In case of error, wait a moment before trying again
                time.sleep(0.5)  # Reduced wait time on error
                
                # Return an error frame if possible, otherwise just continue
                try:
                    error_img = create_error_frame(str(e))
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_img + b'\r\n')
                except:
                    pass
                    
    finally:
        # Make sure to reset the streaming flag when done
        streaming_active = False


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
    # Store the confidence threshold in app config
    app.config['CAT_CONFIDENCE_THRESHOLD'] = float(request.args.get('confidence', '0.5'))
    
    # Get requested FPS (default to 5)
    fps = float(request.args.get('fps', '5'))
    
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


@app.route('/cat_detection_viewer')
def cat_detection_viewer():
    """Endpoint that shows an optimized HTML page to view the cat detection stream in real-time with minimal latency"""
    # Fixed FPS value - can be adjusted here if needed
    fps = request.args.get('fps', '5')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Cat Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 10px;
                background-color: #f5f5f5;
                margin: 0;
            }}
            h1 {{ 
                color: #333; 
                margin-bottom: 15px;
                font-size: 24px;
            }}
            .stream-container {{ 
                max-width: 95%; 
                margin: 0 auto 15px auto; 
                border: 2px solid #333; 
                border-radius: 8px; 
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                background-color: #000;
                position: relative;
            }}
            .stream-container img {{ 
                max-width: 100%; 
                height: auto;
                display: block;
            }}
            .controls {{ 
                margin: 15px auto;
                max-width: 500px;
                background-color: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .control-group {{
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-wrap: wrap;
            }}
            label {{ 
                font-weight: bold;
                margin-right: 8px;
            }}
            .btn-group {{
                margin-top: 10px;
            }}
            button {{ 
                padding: 8px 16px; 
                background: #4CAF50; 
                color: white; 
                border: none;
                border-radius: 4px;
                cursor: pointer;
                margin: 0 4px;
                font-weight: bold;
                font-size: 14px;
            }}
            button:hover {{ background: #45a049; }}
            button.stop {{ background: #f44336; }}
            button.stop:hover {{ background: #d32f2f; }}
            button.start {{ background: #2196F3; }}
            button.start:hover {{ background: #0b7dda; }}
            input[type="range"] {{
                width: 120px;
                margin: 0 10px;
            }}
            input[type="number"] {{ 
                padding: 6px; 
                width: 50px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
            .status-container {{
                display: flex;
                position: absolute;
                top: 10px;
                right: 10px;
                gap: 8px;
            }}
            .status {{
                padding: 4px 8px;
                background-color: rgba(0,0,0,0.6);
                color: white;
                border-radius: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            .cat-detected {{
                background-color: rgba(76, 175, 80, 0.8) !important;
            }}
            .fps-display {{
                position: absolute;
                bottom: 8px;
                left: 8px;
                padding: 4px 8px;
                background-color: rgba(0,0,0,0.6);
                color: white;
                border-radius: 12px;
                font-size: 13px;
            }}
            .processing-time {{
                position: absolute;
                bottom: 8px;
                right: 8px;
                padding: 4px 8px;
                background-color: rgba(0,0,0,0.6);
                color: white;
                border-radius: 12px;
                font-size: 13px;
            }}
            .note {{
                font-style: italic;
                font-size: 13px;
                color: #666;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Real-time Cat Detection</h1>
        
        <div class="stream-container">
            <img src="/stream_cat_detection?fps={fps}" alt="Cat Detection Stream" id="stream">
            <div class="status-container">
                <div class="status" id="status">No cats detected</div>
            </div>
            <div class="fps-display" id="fps">0 FPS</div>
            <div class="processing-time" id="processing-time">0 ms</div>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="fps">Frame Rate:</label>
                <input type="range" id="fps-slider" min="1" max="15" step="1" value="{fps}">
                <input type="number" id="fps" value="{fps}" min="1" max="15" step="1">
            </div>
            
            <div class="btn-group">
                <button onclick="updateSettings()" class="update">Apply Settings</button>
                <button onclick="toggleStream()" class="stop" id="stream-toggle">Pause</button>
            </div>
            
            <div class="note">
                Running at maximum detection sensitivity. Once a cat is detected, its position will be tracked indefinitely.
            </div>
        </div>
        
        <script>
            // Performance optimizations
            const streamImg = document.getElementById('stream');
            const statusEl = document.getElementById('status');
            const fpsEl = document.getElementById('fps');
            const procTimeEl = document.getElementById('processing-time');
            const toggleBtn = document.getElementById('stream-toggle');
            
            let isStreaming = true;
            let frameCount = 0;
            let lastFrameTime = performance.now();
            let lastUrl = '';
            
            // Track FPS with high precision
            let frameTimestamps = [];
            
            // Sync range sliders with number inputs
            document.getElementById('fps-slider').addEventListener('input', function() {{
                document.getElementById('fps').value = this.value;
            }});
            
            document.getElementById('fps').addEventListener('change', function() {{
                document.getElementById('fps-slider').value = this.value;
            }});
            
            // Stream image loaded handler - optimized for performance
            streamImg.addEventListener('load', function() {{
                // Add timestamp to the array
                const now = performance.now();
                frameTimestamps.push(now);
                
                // Only keep the last 10 frames for FPS calculation
                if (frameTimestamps.length > 10) {{
                    frameTimestamps.shift();
                }}
                
                // Calculate FPS from the timestamps
                if (frameTimestamps.length >= 2) {{
                    const timeElapsed = frameTimestamps[frameTimestamps.length - 1] - frameTimestamps[0];
                    const frameCount = frameTimestamps.length - 1;
                    const fps = Math.round((frameCount / timeElapsed) * 1000);
                    fpsEl.textContent = fps + ' FPS';
                }}
            }}, {{ passive: true }}); // Passive event for better performance
            
            // Function to update settings
            function updateSettings() {{
                if (!isStreaming) toggleStream();
                
                const fps = document.getElementById('fps').value;
                
                // Create new stream URL
                const newUrl = `/stream_cat_detection?fps=${{fps}}&_t=${{Date.now()}}`;
                
                // Only update if different from current
                if (newUrl !== lastUrl) {{
                    streamImg.src = newUrl;
                    lastUrl = newUrl;
                    
                    // Reset FPS tracking
                    frameTimestamps = [];
                }}
            }}
            
            // Toggle streaming on/off
            function toggleStream() {{
                if (isStreaming) {{
                    // Pause stream
                    streamImg.dataset.previousSrc = streamImg.src;
                    // Empty 1x1 transparent GIF - much smaller than PNG
                    streamImg.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';
                    toggleBtn.textContent = 'Resume';
                    toggleBtn.classList.remove('stop');
                    toggleBtn.classList.add('start');
                    statusEl.textContent = 'Stream paused';
                }} else {{
                    // Resume stream
                    if (streamImg.dataset.previousSrc) {{
                        streamImg.src = streamImg.dataset.previousSrc;
                    }} else {{
                        updateSettings();
                    }}
                    toggleBtn.textContent = 'Pause';
                    toggleBtn.classList.remove('start');
                    toggleBtn.classList.add('stop');
                }}
                
                isStreaming = !isStreaming;
            }}
            
            // Update processing time display (uses X-Processing-Time header)
            const observer = new MutationObserver(function(mutations) {{
                mutations.forEach(function(mutation) {{
                    if (mutation.type === 'attributes' && mutation.attributeName === 'src') {{
                        // When image src changes, reset processing time display
                        procTimeEl.textContent = 'Processing...';
                    }}
                }});
            }});
            
            // Start observing the image for src attribute changes
            observer.observe(streamImg, {{ attributes: true, attributeFilter: ['src'] }});
            
            // Set up message channel for backend metrics reporting
            if (window.EventSource) {{
                const eventSource = new EventSource('/stream_metrics');
                eventSource.onmessage = function(event) {{
                    try {{
                        const data = JSON.parse(event.data);
                        if (data.processing_time) {{
                            procTimeEl.textContent = data.processing_time + ' ms';
                        }}
                        if (data.cat_detected !== undefined) {{
                            if (data.cat_detected) {{
                                statusEl.textContent = `Cat detected`;
                                statusEl.classList.add('cat-detected');
                            }} else {{
                                statusEl.textContent = 'No cats detected';
                                statusEl.classList.remove('cat-detected');
                            }}
                        }}
                    }} catch (e) {{
                        console.error('Error parsing metrics:', e);
                    }}
                }};
            }}
        </script>
    </body>
    </html>
    """
    return html


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
    print("Initializing laser-turret camera system as API backend...")

    # Register cleanup function to run on exit
    import atexit

    atexit.register(cleanup)

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("Received shutdown signal, cleaning up...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set up camera
    if setup_camera() is None:
        print("Warning: Camera initialization failed. Starting server anyway.")
        print("You can try to initialize the camera later using the /reset endpoint.")

    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)  # Port 5000 is used by default
    # Endpoints are: /status, /capture, /info, /reset, /detect_cat, /detect_cat_json

if __name__ == "__main__":
    main()

# Start with "python3 /home/mason/camerastuff/main.py"

# Cat Detection:
# Start the application with: python3 /home/mason/camerastuff/main.py
# Access the cat detection endpoint at http://<your-ip>:5000/detect_cat
# Get detection results as JSON at http://<your-ip>:5000/detect_cat_json
