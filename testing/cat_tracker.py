import time
import json
import threading
import queue
import requests
import logging
from .servo_controller import DualServoController

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CatTracker:
    """
    Integrates cat detection results with the servo controller to track cats in real-time.
    This class handles:
    1. Getting detection results from the backend
    2. Processing detection coordinates
    3. Controlling the servo to target detected cats
    """
    
    def __init__(self, backend_url, pan_pin=17, tilt_pin=27, poll_interval=0.2):
        """
        Initialize the cat tracker with backend connection and servo controller
        
        Args:
            backend_url: URL of the backend service (e.g. "http://192.168.1.100:5001")
            pan_pin: GPIO pin for pan servo
            tilt_pin: GPIO pin for tilt servo
            poll_interval: How often to check for new detections (seconds)
        """
        self.backend_url = backend_url
        self.poll_interval = poll_interval
        
        # Initialize servo controller
        self.servo = DualServoController(pan_pin=pan_pin, tilt_pin=tilt_pin)
        
        # Queue for passing detection data to the servo controller
        self.detection_queue = queue.Queue(maxsize=10)
        
        # Tracking control
        self.tracking = False
        self._tracking_thread = None
        self.last_detection_time = 0
        
    def start_tracking(self):
        """Start the cat tracking system"""
        if self._tracking_thread and self._tracking_thread.is_alive():
            logger.warning("Tracking already active")
            return
            
        self.tracking = True
        
        # Start the servo controller's tracking thread
        self.servo.start_tracking(detection_queue=self.detection_queue, 
                                 poll_interval=self.poll_interval)
        
        # Start the detection polling thread
        self._tracking_thread = threading.Thread(target=self._poll_detections)
        self._tracking_thread.daemon = True
        self._tracking_thread.start()
        
        logger.info("Cat tracking started")
        
    def stop_tracking(self):
        """Stop the cat tracking system"""
        self.tracking = False
        
        # Stop the servo controller tracking
        self.servo.stop_tracking()
        
        # Wait for polling thread to finish
        if self._tracking_thread and self._tracking_thread.is_alive():
            self._tracking_thread.join(timeout=2.0)
        
        logger.info("Cat tracking stopped")
        
    def _poll_detections(self):
        """Thread function to continuously poll for new cat detections"""
        logger.info("Starting detection polling thread")
        
        while self.tracking:
            try:
                # Get the latest detection results from backend
                response = requests.get(
                    f"{self.backend_url}/detect_cat_json",
                    timeout=2.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    detection_result = data.get("detection_result", {})
                    
                    # Check if a cat was detected
                    if detection_result.get("detected", False):
                        # Get the first (or only) bounding box
                        bounding_boxes = detection_result.get("bounding_boxes", [])
                        if bounding_boxes:
                            # Get the first bounding box
                            box = bounding_boxes[0]
                            
                            # Log detection for testing
                            center_x = box['x'] + (box['width'] // 2)
                            center_y = box['y'] + (box['height'] // 2)
                            logger.info(f"Cat detected: center=({center_x}, {center_y}), conf={box.get('confidence', 0):.2f}")
                            
                            # Put in queue for servo controller to process
                            try:
                                if not self.detection_queue.full():
                                    self.detection_queue.put_nowait(box)
                                    self.last_detection_time = time.time()
                            except queue.Full:
                                # If queue is full, drop this detection
                                pass
                    else:
                        # No cat detected in this frame
                        if time.time() - self.last_detection_time > 5.0:
                            logger.debug("No cats detected for 5 seconds")
                
            except requests.RequestException as e:
                logger.error(f"Error connecting to backend: {e}")
            except Exception as e:
                logger.error(f"Error in detection polling: {e}")
                
            # Sleep before next poll
            time.sleep(self.poll_interval)
            
        logger.info("Detection polling thread stopped")
        
    def test_corners(self):
        """Test servo movement by moving to the four corners"""
        self.servo.scan_corners(pause_time=1.5)
        
    def center(self):
        """Center the servos"""
        self.servo.center()
        
    def cleanup(self):
        """Clean up resources"""
        self.stop_tracking()
        self.servo.cleanup()


# Example usage for testing
if __name__ == "__main__":
    # Backend URL - replace with your actual backend server address
    BACKEND_URL = "http://192.168.1.100:5001"
    
    # Create tracker
    tracker = CatTracker(
        backend_url=BACKEND_URL,
        pan_pin=17,   # Use your actual GPIO pin
        tilt_pin=27   # Use your actual GPIO pin
    )
    
    try:
        # Center to start
        tracker.center()
        time.sleep(1)
        
        # Test corner movement
        print("Testing corner movement...")
        tracker.test_corners()
        time.sleep(1)
        
        # Start tracking
        print("Starting cat tracking. Press Ctrl+C to stop.")
        tracker.start_tracking()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        # Clean up
        tracker.cleanup()