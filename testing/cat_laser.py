#!/usr/bin/env python3
"""
Cat Laser Tracking System

This script integrates the cat detection system with servo control to point a laser
at a detected cat. The system captures images, detects cats, and moves the servos
to point the laser at the cat's position.

Usage:
    python cat_laser.py [options]

Options:
    --backend-url URL   URL of the backend server (default: http://127.0.0.1:5001)
    --frontend-url URL  URL of the frontend server (default: http://127.0.0.1:5000)
    --pan-pin PIN       GPIO pin for the pan servo (default: 18)
    --tilt-pin PIN      GPIO pin for the tilt servo (default: 17)
    --poll-rate RATE    How often to check for cats in seconds (default: 0.2)
    --debug             Enable debug logging
"""

import argparse
import time
import threading
import requests
import logging
import sys
import os
import signal
import queue
import numpy as np

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import servo controller - direct import since frontend is not a package
from servo_controller import DualServoController

# Import shared configuration
from shared.servo_config import (
    CAMERA_CORNERS, 
    STANDARD_CENTER, 
    SERVO_LIMITS, 
    DEFAULT_GPIO_PINS,
    MOVEMENT_SETTINGS,
    CAMERA_SETTINGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

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
        self.servo = DualServoController(pan_pin=pan_pin, tilt_pin=tilt_pin)
        
        # Set servo movement parameters
        self.servo.pan_step = pan_speed
        self.servo.tilt_step = tilt_speed
        self.servo.movement_delay = move_delay
        
        self.tracking_active = False
        self.tracking_thread = None
        
        # Custom coordinate mapping with calibrated corners
        self.map_coordinates_to_angles = self.calibrated_map_coordinates_to_angles
        
        # Calculate camera center point from the corners
        self.camera_center = {
            'x': self.image_width / 2,
            'y': self.image_height / 2
        }
        
        # Queue for tracking detected cats - initialize as a queue.Queue instead of a list
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
                            
                            # Calculate center of bounding box
                            center_x = box["x"] + (box["width"] // 2)
                            center_y = box["y"] + (box["height"] // 2)
                            
                            # Log detection
                            logger.info(f"Cat detected at ({center_x}, {center_y}), confidence: {box.get('confidence', 0):.2f}")
                            
                            # Use our calibrated mapping function to convert coordinates to angles
                            pan_angle, tilt_angle = self.calibrated_map_coordinates_to_angles(center_x, center_y)
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
                                pass  # Skip if queue is full
                    else:
                        # No cat detected
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
        self.servo.cleanup()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cat Laser Tracking System")
    
    parser.add_argument("--camera-width", 
                        type=int, 
                        default=CAMERA_SETTINGS['width'],
                        help="Camera width in pixels")
    parser.add_argument("--camera-height", 
                        type=int, 
                        default=CAMERA_SETTINGS['height'],
                        help="Camera height in pixels")
    parser.add_argument("--camera-fps", 
                        type=int, 
                        default=CAMERA_SETTINGS['fps'],
                        help="Camera frames per second")
    parser.add_argument("--backend-url", 
                        default="http://127.0.0.1:5001",
                        help="URL of the backend server")
    parser.add_argument("--frontend-url",
                        default="http://127.0.0.1:5000",
                        help="URL of the frontend server")
    parser.add_argument("--pan-pin", 
                        type=int, 
                        default=DEFAULT_GPIO_PINS['pan_pin'], 
                        help="GPIO pin for pan servo")
    parser.add_argument("--tilt-pin", 
                        type=int, 
                        default=DEFAULT_GPIO_PINS['tilt_pin'], 
                        help="GPIO pin for tilt servo")
    parser.add_argument("--poll-rate", 
                        type=float, 
                        default=0.2, 
                        help="How often to check for cats (seconds)")
    parser.add_argument("--pan-speed", 
                        type=float, 
                        default=MOVEMENT_SETTINGS['pan_step'], 
                        help="Pan servo movement speed (degrees per move)")
    parser.add_argument("--tilt-speed", 
                        type=float, 
                        default=MOVEMENT_SETTINGS['tilt_step'], 
                        help="Tilt servo movement speed (degrees per move)")
    parser.add_argument("--move-delay", 
                        type=float, 
                        default=MOVEMENT_SETTINGS['movement_delay'], 
                        help="Delay between servo movements (seconds)")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Set default GPIO pins to match test_both_servos.py
    pan_pin = args.pan_pin 
    tilt_pin = args.tilt_pin
    
    # Log configuration
    logger.info("Starting Cat Laser Tracking System")
    logger.info(f"Backend URL: {args.backend_url}")
    logger.info(f"Frontend URL: {args.frontend_url}")
    logger.info(f"Pan servo pin: {pan_pin}")
    logger.info(f"Tilt servo pin: {tilt_pin}")
    logger.info(f"Poll rate: {args.poll_rate} seconds")
    logger.info(f"Servo speed settings - Pan: {args.pan_speed} deg/move, Tilt: {args.tilt_speed} deg/move, Delay: {args.move_delay}s")

    # Log camera configuration
    logger.info(f"Camera resolution: {args.camera_width}x{args.camera_height} @ {args.camera_fps}fps")
    
    # Create tracker with the specified parameters
    tracker = CatLaserTracker(
        backend_url=args.backend_url,
        frontend_url=args.frontend_url,
        pan_pin=pan_pin,
        tilt_pin=tilt_pin,
        poll_rate=args.poll_rate,
        image_width=args.camera_width,
        image_height=args.camera_height,
        pan_speed=args.pan_speed,
        tilt_speed=args.tilt_speed,
        move_delay=args.move_delay
    )
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        tracker.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Center servos to start
        logger.info("Centering servos")
        tracker.servo.center()
        time.sleep(1)
        
        # Test servo range
        logger.info("Testing servo range")
        tracker.test_corners()
        time.sleep(1)
        
        # Start tracking
        logger.info("Starting cat laser tracking")
        tracker.start()
        
        # Keep the main thread running
        logger.info("System running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        # Clean up
        tracker.cleanup()


if __name__ == "__main__":
    main()