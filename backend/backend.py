import io
import os
import time
import signal
import sys
import threading
import json
from datetime import datetime

# Add the parent directory to the Python path so we can import from shared
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, Response, jsonify, send_file
from shared.cat_detector import detect_cat

app = Flask(__name__)

# Queue for processing images
processing_queue = []
processing_lock = threading.Lock()
processing_thread = None
processing_active = False

# Set this to your Raspberry Pi's IP or hostname
FRONTEND_IP = "192.168.2.3"  # Change to your Raspberry Pi's IP address

@app.route("/status", methods=["GET"])
def api_status():
    """API endpoint to check if the backend service is running"""
    return jsonify({
        "status": "online",
        "message": "Backend cat detection service is running"
    })


@app.route("/process_image", methods=["POST"])
def api_process_image():
    """API endpoint to receive an image from the frontend and process it with cat detection"""
    try:
        start_time = time.time()
        
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get image from request
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Get confidence parameter (use default if not provided)
        confidence = None
        if 'confidence' in request.args:
            try:
                confidence = float(request.args.get('confidence'))
            except ValueError:
                pass
        
        # Get resize parameter
        resize_output = request.args.get('resize_output', 'false').lower() == 'true'
        
        # Process image with cat detection
        annotated_image, detection_result = detect_cat(image_data, confidence, resize_output)
        
        # Create response with the annotated image
        response = Response(
            annotated_image,
            mimetype="image/jpeg"
        )
        
        # Add detection results as headers
        response.headers['X-Cat-Detected'] = str(detection_result['detected'])
        response.headers['X-Cat-Count'] = str(detection_result['count'])
        response.headers['X-Processing-Time'] = str(detection_result.get('processing_time_ms', 0))
        
        # Log processing time and result
        processing_time = time.time() - start_time
        print(f"Image processed in {processing_time*1000:.1f}ms, "
              f"Cat detected: {detection_result['detected']}, "
              f"Count: {detection_result['count']}")
        
        return response
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/process_image_json", methods=["POST"])
def api_process_image_json():
    """API endpoint to process an image and return JSON results for servo control"""
    try:
        start_time = time.time()
        
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Get image from request
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Get confidence parameter (use default if not provided)
        confidence = None
        if 'confidence' in request.args:
            try:
                confidence = float(request.args.get('confidence'))
            except ValueError:
                pass
        
        # Process image with cat detection
        annotated_image, detection_result = detect_cat(image_data, confidence)
        
        # Add processing time to the result
        processing_time = time.time() - start_time
        detection_result['processing_time_ms'] = round(processing_time * 1000)
        
        # Log detection
        if detection_result['detected']:
            boxes = detection_result.get('bounding_boxes', [])
            if boxes:
                box = boxes[0]  # Get first box
                print(f"Cat detected! Position: ({box['x']}, {box['y']}), "
                      f"Size: {box['width']}x{box['height']}, "
                      f"Confidence: {box.get('confidence', 0):.2f}")
        
        return jsonify(detection_result)
        
    except Exception as e:
        print(f"Error processing image for JSON response: {e}")
        return jsonify({"error": str(e), "detected": False}), 500


@app.route('/stream_process_queue', methods=['POST'])
def stream_process_queue():
    """
    Endpoint for frontend to queue an image for background processing.
    This allows the frontend to continue capturing images without waiting for processing.
    """
    global processing_queue, processing_active
    
    try:
        # Check if image was sent
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        # Get image from request
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Get queue ID from parameters or generate one
        queue_id = request.args.get('queue_id', str(int(time.time() * 1000)))
        
        # Add to processing queue with lock for thread safety
        with processing_lock:
            # Limit queue size to prevent memory issues
            if len(processing_queue) >= 10:
                # Remove oldest item
                processing_queue.pop(0)
            
            # Add new image to queue
            processing_queue.append({
                'id': queue_id,
                'image_data': image_data,
                'timestamp': time.time(),
                'processed': False,
                'result': None
            })
        
        # Make sure processing thread is running
        ensure_processing_thread()
        
        return jsonify({
            "status": "queued",
            "queue_id": queue_id,
            "position": len(processing_queue)
        })
        
    except Exception as e:
        print(f"Error queueing image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_processed_image/<queue_id>', methods=['GET'])
def get_processed_image(queue_id):
    """
    Retrieve a processed image by its queue ID.
    This allows the frontend to poll for results after submitting images.
    """
    global processing_queue
    
    try:
        # Look for the image in the processing queue
        with processing_lock:
            for item in processing_queue:
                if item['id'] == queue_id:
                    if item['processed'] and item['result']:
                        # Return the processed image
                        annotated_image = item['result'].get('annotated_image')
                        detection_result = item['result'].get('detection_result', {})
                        
                        if annotated_image:
                            response = Response(
                                annotated_image,
                                mimetype="image/jpeg"
                            )
                            
                            # Add detection results as headers
                            response.headers['X-Cat-Detected'] = str(detection_result.get('detected', False))
                            response.headers['X-Cat-Count'] = str(detection_result.get('count', 0))
                            response.headers['X-Processing-Time'] = str(detection_result.get('processing_time_ms', 0))
                            
                            # Remove item from queue to save memory
                            processing_queue.remove(item)
                            
                            return response
                        else:
                            return jsonify({"error": "Processed image not available"}), 500
                    else:
                        # Image is still being processed
                        return jsonify({
                            "status": "processing",
                            "message": "Image is still being processed"
                        }), 202
        
        # If we get here, the queue_id wasn't found
        return jsonify({"error": "Image not found in queue"}), 404
        
    except Exception as e:
        print(f"Error retrieving processed image: {e}")
        return jsonify({"error": str(e)}), 500


def process_queue_worker():
    """Background worker thread to process images in the queue"""
    global processing_queue, processing_active
    
    processing_active = True
    print("Background processing thread started")
    
    try:
        while processing_active:
            # Check if there are images to process
            with processing_lock:
                unprocessed_items = [item for item in processing_queue if not item['processed']]
            
            if unprocessed_items:
                # Process the oldest unprocessed item first
                item = unprocessed_items[0]
                
                try:
                    # Process the image
                    print(f"Processing queued image {item['id']}")
                    annotated_image, detection_result = detect_cat(item['image_data'])
                    
                    # Update the item with results
                    with processing_lock:
                        # Find the item again (it might have been removed)
                        for queue_item in processing_queue:
                            if queue_item['id'] == item['id']:
                                queue_item['processed'] = True
                                queue_item['result'] = {
                                    'annotated_image': annotated_image,
                                    'detection_result': detection_result
                                }
                                break
                    
                    # Log processing
                    print(f"Processed queued image {item['id']}, "
                          f"Cat detected: {detection_result['detected']}, "
                          f"Count: {detection_result['count']}")
                
                except Exception as e:
                    print(f"Error processing queued image {item['id']}: {e}")
                    
                    # Mark as processed with error
                    with processing_lock:
                        for queue_item in processing_queue:
                            if queue_item['id'] == item['id']:
                                queue_item['processed'] = True
                                queue_item['result'] = {'error': str(e)}
                                break
            else:
                # No images to process, sleep briefly
                time.sleep(0.1)
            
            # Clean up old items
            current_time = time.time()
            with processing_lock:
                # Remove items older than 5 minutes
                processing_queue = [item for item in processing_queue 
                                    if (current_time - item['timestamp']) < 300]
        
    except Exception as e:
        print(f"Error in processing thread: {e}")
    
    print("Background processing thread stopped")
    processing_active = False


def ensure_processing_thread():
    """Ensure the background processing thread is running"""
    global processing_thread, processing_active
    
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_queue_worker)
        processing_thread.daemon = True
        processing_thread.start()


def cleanup():
    """Clean up resources before exiting"""
    global processing_active
    
    print("Shutting down backend...")
    processing_active = False
    
    # Wait for processing thread to finish
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=1.0)


def main():
    print("Initializing backend cat detection service...")

    # Register cleanup function
    import atexit
    atexit.register(cleanup)

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("Received shutdown signal, cleaning up...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the processing thread
    ensure_processing_thread()

    # Run the Flask app - using port 5001 for backend
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()

# Start with "python3 /home/mason/camerastuff/backend.py"