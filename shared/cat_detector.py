import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import time

# Define paths for YOLO model files
MODEL_DIR = Path(__file__).parent / "models" / "yolo"
WEIGHTS_PATH = MODEL_DIR / "yolov4-tiny.weights"
CONFIG_PATH = MODEL_DIR / "yolov4-tiny.cfg"
CLASSES_PATH = MODEL_DIR / "coco.names"

# Class ID for cats in COCO dataset
CAT_CLASS_ID = 15  # (0-indexed COCO class for 'cat')

# Global variables for tracking
last_frame = None
motion_threshold = 15  # Motion detection sensitivity (lower is more sensitive)
last_cat_boxes = []
box_persistence_time = float('inf')  # Infinite persistence - boxes never disappear once a cat is detected
last_cat_detection_time = 0
minimum_confidence = 0.1  # Minimum confidence threshold - very low for high sensitivity

# Keep model in memory for faster inference
global_model = None
global_output_layers = None
global_classes = None

# Optimal processing resolution
PROCESS_WIDTH = 416  # Width to process with YOLO (lower = faster)
PROCESS_HEIGHT = 416  # Height to process with YOLO (lower = faster)

def download_model_files():
    """Download YOLOv4-tiny model files if they don't exist"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Download class names
    if not CLASSES_PATH.exists():
        print("Downloading COCO class names...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
            CLASSES_PATH
        )
    
    # Download config file
    if not CONFIG_PATH.exists():
        print("Downloading YOLOv4-tiny config...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            CONFIG_PATH
        )
    
    # Download weights file
    if not WEIGHTS_PATH.exists():
        print("Downloading YOLOv4-tiny weights (this may take a moment)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
            WEIGHTS_PATH
        )

def load_yolo_model():
    """Load YOLOv4-tiny model and return the network"""
    global global_model, global_output_layers, global_classes
    
    # Return cached model if available
    if global_model is not None and global_output_layers is not None and global_classes is not None:
        return global_model, global_output_layers, global_classes
    
    # Make sure model files exist
    download_model_files()
    
    # Load class names
    with open(CLASSES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Load the network
    net = cv2.dnn.readNet(str(WEIGHTS_PATH), str(CONFIG_PATH))
    
    # Use CUDA if available
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.dnn.DNN_TARGET_CPU)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Cache the model
    global_model = net
    global_output_layers = output_layers
    global_classes = class_names
    
    return net, output_layers, class_names

def detect_motion(current_frame, previous_frame, scale_factor=0.5):
    """
    Detect motion between two frames with optimization for speed
    Returns: motion detected (boolean), motion frame (for visualization)
    """
    if previous_frame is None:
        return False, None
    
    # Resize frames for faster motion detection
    h, w = current_frame.shape[:2]
    small_h, small_w = int(h * scale_factor), int(w * scale_factor)
    
    small_current = cv2.resize(current_frame, (small_w, small_h))
    small_previous = cv2.resize(previous_frame, (small_w, small_h))
    
    # Convert frames to grayscale
    gray_current = cv2.cvtColor(small_current, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(small_previous, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray_current, gray_previous)
    
    # Apply threshold to difference
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Count white pixels (changed pixels)
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = small_w * small_h
    change_percent = (white_pixels / total_pixels) * 100
    
    # If change exceeds threshold, motion is detected
    motion_detected = change_percent > 0.5  # Threshold percentage
    
    return motion_detected, None  # Skip creating motion frame to save time

def calculate_box_overlap(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes"""
    # Extract coordinates
    x1_1, y1_1 = box1['x'], box1['y']
    x2_1, y2_1 = box1['x'] + box1['width'], box1['y'] + box1['height']
    
    x1_2, y1_2 = box2['x'], box2['y']
    x2_2, y2_2 = box2['x'] + box2['width'], box2['y'] + box2['height']
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Return IoU
    return intersection_area / union_area if union_area > 0 else 0.0

def track_cats(current_detections, last_detections, current_time):
    """Track cats across frames to maintain consistent detection"""
    global last_cat_detection_time
    
    if len(current_detections) > 0:
        # If cats are currently detected, update the last detection time
        last_cat_detection_time = current_time
        return current_detections
        
    # If no current detections but we have previous ones and we're within persistence window
    if len(last_detections) > 0 and (current_time - last_cat_detection_time) < box_persistence_time:
        # Keep the previous detections but reduce their confidence slightly
        for box in last_detections:
            if 'confidence' in box:
                box['confidence'] = max(0.0, box['confidence'] - 0.05)
                
            # Mark as tracked (not directly detected)
            box['tracked'] = True
        
        return last_detections
    
    # Otherwise, no cats to track
    return []

def resize_for_display(image, max_width=800):
    """Resize image for efficient browser display"""
    h, w = image.shape[:2]
    
    # Only resize if image is larger than max_width
    if w > max_width:
        scale = max_width / w
        new_width = max_width
        new_height = int(h * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    return image

def detect_cat(image_data, confidence_threshold=None, resize_output=True):
    """
    Detect cats in the given image data and annotate the image with bounding boxes.
    Only displays the single highest-confidence cat detection with maximum sensitivity.
    Annotations persist indefinitely once a cat is detected.
    
    Args:
        image_data: Image data as bytes or BytesIO object
        confidence_threshold: Parameter is ignored, always uses minimum_confidence
        resize_output: Whether to resize output for browser display (default: True)
    
    Returns:
        tuple: (annotated_image, detection_result)
            - annotated_image: The image with bounding boxes drawn around detected cats
            - detection_result: Dict with detection information
    """
    global last_frame, last_cat_boxes, last_cat_detection_time
    current_time = time.time()
    start_time = current_time  # For timing performance
    
    # Always use minimum confidence for highest sensitivity
    confidence_threshold = minimum_confidence
    
    # Convert image data to OpenCV format
    if hasattr(image_data, 'read'):
        # If it's a file-like object (BytesIO), get the bytes
        image_data.seek(0)
        img_array = np.frombuffer(image_data.read(), np.uint8)
        image_data.seek(0)  # Reset the stream position
    else:
        # If it's already bytes
        img_array = np.frombuffer(image_data, np.uint8)
    
    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image data")
    
    # Calculate original dimensions
    original_h, original_w = img.shape[:2]
    
    # Make a copy of the original image to annotate
    annotated_img = img.copy()
    motion_detected = False
    
    # Check for motion if we have a previous frame (use fast motion detection)
    if last_frame is not None:
        motion_detected, _ = detect_motion(img, last_frame, scale_factor=0.25)
        
        # Always be very sensitive when motion is detected
        if motion_detected:
            confidence_threshold = min(confidence_threshold, 0.05)
    
    # Always store current frame for next comparison
    last_frame = img.copy()
    
    try:
        # Load YOLO model (uses cached model)
        net, output_layers, classes = load_yolo_model()
        
        # Prepare the image for YOLO detection - resize to optimal size for processing
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (PROCESS_WIDTH, PROCESS_HEIGHT), swapRB=True, crop=False)
        
        # Set the input to the network
        net.setInput(blob)
        
        # Run forward pass
        outputs = net.forward(output_layers)
        
        # Process the outputs
        cat_detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # If we detected a cat with sufficient confidence
                if class_id == CAT_CLASS_ID and confidence > confidence_threshold:
                    # Convert coordinates from network processing size to original image size
                    center_x = int(detection[0] * original_w)
                    center_y = int(detection[1] * original_h)
                    w = int(detection[2] * original_w)
                    h = int(detection[3] * original_h)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    cat_detections.append({
                        "x": max(0, x),
                        "y": max(0, y),
                        "width": w,
                        "height": h,
                        "confidence": float(confidence),
                        "tracked": False
                    })
        
        # Apply tracking to maintain cat detection consistency
        # This will now keep the cat box indefinitely due to infinite persistence time
        cat_detections = track_cats(cat_detections, last_cat_boxes, current_time)
        
        # Keep only the single highest-confidence cat detection
        if len(cat_detections) > 1:
            # Sort by confidence (highest first) and keep only the first one
            cat_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            cat_detections = [cat_detections[0]]  # Keep only the best detection
        
        # Update for next frame - keep track of this single detection
        if cat_detections:
            last_cat_boxes = cat_detections.copy()
            # If cat is detected in this frame, update the detection time
            if not cat_detections[0].get('tracked', False):
                last_cat_detection_time = current_time
        
        # Draw boxes on the image (now we'll have at most one box)
        for box in cat_detections:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            confidence = box.get('confidence', 0.0)
            tracked = box.get('tracked', False)
            
            # Use different color for tracked boxes
            color = (0, 255, 0) if not tracked else (0, 255, 255)  # Green for detected, Yellow for tracked
            
            # Draw the bounding box
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 4)  # Increased thickness from 2 to 4
            
            # Draw label
            label_text = f"Cat: {confidence:.2f}"
            if tracked:
                # Add time since last actual detection
                seconds_since_detection = current_time - last_cat_detection_time
                if seconds_since_detection > 5:
                    label_text = f"Last seen: {int(seconds_since_detection)}s ago"
                else:
                    label_text += " (tracked)"
            cv2.putText(annotated_img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add motion indication if detected (small text to reduce rendering overhead)
        if motion_detected:
            cv2.putText(annotated_img, "Motion", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        cv2.putText(annotated_img, f"{processing_time*1000:.0f}ms", (annotated_img.shape[1]-70, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Create result
        result = {
            "detected": len(cat_detections) > 0,
            "count": len(cat_detections),  # This will now be either 0 or 1
            "bounding_boxes": cat_detections,
            "processing_time_ms": round(processing_time * 1000)
        }
        
        # Resize output image for browser if needed (reduces bandwidth and improves latency)
        if resize_output:
            annotated_img = resize_for_display(annotated_img)
        
    except Exception as e:
        print(f"Error during cat detection: {e}")
        result = {"detected": False, "count": 0, "error": str(e), "bounding_boxes": []}
    
    # Compress with reduced quality for better streaming
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    _, img_encoded = cv2.imencode('.jpg', annotated_img, encode_params)
    
    return img_encoded.tobytes(), result