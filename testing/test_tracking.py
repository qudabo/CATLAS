#!/usr/bin/env python3
"""
Cat Detection Coordinate Tester

This script connects to the backend service and prints the coordinates
of detected cats in the terminal. Use this for testing the detection
system without needing to connect actual servos.

Usage:
    python test_tracking.py [--backend-url URL]

Options:
    --backend-url URL   URL of the backend detection service (default: http://localhost:5001)
"""

import os
import sys
import time
import json
import argparse
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cat Detection Coordinate Tester")
    parser.add_argument("--backend-url", 
                        default="http://localhost:5001",
                        help="URL of the backend detection service")
    return parser.parse_args()

def print_box(box, frame_width=800, frame_height=600):
    """
    Print a visualization of the bounding box in the terminal
    """
    # Calculate center coordinates
    center_x = box['x'] + (box['width'] // 2)
    center_y = box['y'] + (box['height'] // 2)
    
    # Scale for terminal display
    term_width = 60  # characters
    term_height = 20  # lines
    
    x_ratio = term_width / frame_width
    y_ratio = term_height / frame_height
    
    term_x = int(center_x * x_ratio)
    term_y = int(center_y * y_ratio)
    
    # Create a simple visualization
    display = []
    for y in range(term_height):
        line = []
        for x in range(term_width):
            if x == term_x and y == term_y:
                line.append("X")  # Mark center point
            else:
                line.append(".")
        display.append("".join(line))
    
    # Print the visualization
    print("\033c", end="")  # Clear terminal
    print(f"Cat Detection - {datetime.now().strftime('%H:%M:%S')}")
    print(f"Box: x={box['x']}, y={box['y']}, width={box['width']}, height={box['height']}")
    print(f"Center Point: ({center_x}, {center_y})")
    print(f"Confidence: {box.get('confidence', 0):.2f}")
    print("\n".join(display))

def main():
    args = parse_arguments()
    backend_url = args.backend_url
    
    print(f"Cat Detection Coordinate Tester")
    print(f"Backend URL: {backend_url}")
    print(f"Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        last_detection_time = 0
        update_interval = 0.2  # seconds
        
        while True:
            try:
                # Get detection results from backend
                response = requests.get(
                    f"{backend_url}/detect_cat_json",
                    timeout=2.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    detection_result = data.get("detection_result", {})
                    
                    # Check if a cat was detected
                    if detection_result.get("detected", False):
                        # Get bounding boxes
                        bounding_boxes = detection_result.get("bounding_boxes", [])
                        if bounding_boxes:
                            box = bounding_boxes[0]  # Get the first detection
                            
                            # Print coordinates and visualization
                            print_box(box)
                            last_detection_time = time.time()
                    else:
                        # No cat detected
                        if time.time() - last_detection_time > 5.0:
                            print("\033c", end="")  # Clear terminal
                            print(f"No cats detected - {datetime.now().strftime('%H:%M:%S')}")
                            print("Waiting for detection...")
                            last_detection_time = time.time()  # Reset to avoid spam
                else:
                    print(f"Error: Received status code {response.status_code} from backend")
                    time.sleep(1)  # Longer delay on error
                    
            except requests.RequestException as e:
                print(f"Connection error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
                
            # Sleep briefly between polls
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    
if __name__ == "__main__":
    main()