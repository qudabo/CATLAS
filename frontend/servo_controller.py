import pigpio
import time
import json
import threading
import logging
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import shared configuration
from shared.servo_config import (
    CAMERA_CORNERS, 
    STANDARD_CENTER, 
    SERVO_LIMITS, 
    DEFAULT_GPIO_PINS,
    MOVEMENT_SETTINGS,
    CAMERA_SETTINGS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualServoController:
    def __init__(self, pan_pin, tilt_pin, frequency=50, 
                 image_width=CAMERA_SETTINGS['width'], 
                 image_height=CAMERA_SETTINGS['height']):
        self.frequency = frequency
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        
        # Store image dimensions for coordinate mapping
        self.image_width = image_width
        self.image_height = image_height
        
        # Initialize tracking state
        self.tracking_active = False
        self._tracking_thread = None
        self.last_detection = None
        
        # Store current servo positions
        self.current_pan = STANDARD_CENTER['pan']
        self.current_tilt = STANDARD_CENTER['tilt']
        
        # Movement speed control (degrees per move)
        self.pan_step = MOVEMENT_SETTINGS['pan_step']
        self.tilt_step = MOVEMENT_SETTINGS['tilt_step']
        # Movement delay parameter
        self.movement_delay = MOVEMENT_SETTINGS['movement_delay']
        
        # Servo limits (from shared configuration)
        self.pan_min = SERVO_LIMITS['pan_min']
        self.pan_max = SERVO_LIMITS['pan_max']
        self.tilt_min = SERVO_LIMITS['tilt_min']
        self.tilt_max = SERVO_LIMITS['tilt_max']

        # Initialize pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon")

        # Set servo pins as outputs
        self.pi.set_mode(self.pan_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.tilt_pin, pigpio.OUTPUT)
        
        # Add flags for tracking continuous movement
        self.last_movement_time = 0
        self.pwm_active = False
        self.continuous_timeout = 0.5  # Time in seconds before stopping PWM after last movement
        
        logger.info(f"Initialized servo controller with pins: pan={pan_pin}, tilt={tilt_pin}")
        logger.info(f"Servo limits: pan={self.pan_min}-{self.pan_max}, tilt={self.tilt_min}-{self.tilt_max}")
    
    def angle_to_pulse_width(self, angle):
        """Convert an angle (0-180) to pulse width (500-2500 microseconds) for servo control"""
        min_pulse = 500   # Minimum pulse width in microseconds
        max_pulse = 2500  # Maximum pulse width in microseconds
        # Ensure angle is within 0-180 range
        angle = max(0, min(180, angle))
        pulse = min_pulse + ((angle / 180.0) * (max_pulse - min_pulse))
        return int(pulse)
        
    def move_pan(self, angle):
        angle = max(self.pan_min, min(self.pan_max, angle))
        pulse_width = self.angle_to_pulse_width(angle)
        self.pi.set_servo_pulsewidth(self.pan_pin, pulse_width)
        time.sleep(self.movement_delay)  # Reduced for more responsive tracking
        self.pi.set_servo_pulsewidth(self.pan_pin, 0)
        self.current_pan = angle
        
    def move_tilt(self, angle):
        angle = max(self.tilt_min, min(self.tilt_max, angle))
        pulse_width = self.angle_to_pulse_width(angle)
        self.pi.set_servo_pulsewidth(self.tilt_pin, pulse_width)
        time.sleep(self.movement_delay)  # Reduced for more responsive tracking
        self.pi.set_servo_pulsewidth(self.tilt_pin, 0)
        self.current_tilt = angle

    def move(self, pan_angle, tilt_angle):
        """Move both servos simultaneously with minimal delay for continuous movement"""
        # Clamp angles to valid ranges
        pan_angle = max(self.pan_min, min(self.pan_max, pan_angle))
        tilt_angle = max(self.tilt_min, min(self.tilt_max, tilt_angle))
        
        # Calculate pulse widths for both servos
        pan_pulse = self.angle_to_pulse_width(pan_angle)
        tilt_pulse = self.angle_to_pulse_width(tilt_angle)
        
        # Set both servo pulses without any delay
        self.pi.set_servo_pulsewidth(self.pan_pin, pan_pulse)
        self.pi.set_servo_pulsewidth(self.tilt_pin, tilt_pulse)
        self.pwm_active = True
        
        # Update current positions
        self.current_pan = pan_angle
        self.current_tilt = tilt_angle
    
    def stop_pwm(self):
        """Stop PWM signals to both servos"""
        if self.pwm_active:
            self.pi.set_servo_pulsewidth(self.pan_pin, 0)
            self.pi.set_servo_pulsewidth(self.tilt_pin, 0)
            self.pwm_active = False

    def move_pan_tilt(self, pan_angle, tilt_angle):
        """Legacy method name for compatibility"""
        self.move(pan_angle, tilt_angle)
        
    def center(self):
        self.move(STANDARD_CENTER['pan'], STANDARD_CENTER['tilt'])
    
    def update_image_dimensions(self, width, height):
        """Update the image dimensions for coordinate mapping"""
        self.image_width = width
        self.image_height = height
        logger.info(f"Image dimensions updated: {width}x{height}")
        
    def map_coordinates_to_angles(self, x, y):
        """
        Map image coordinates (x, y) to servo angles using bilinear interpolation
        
        Args:
            x: X coordinate in the image (0 to image_width)
            y: Y coordinate in the image (0 to image_height)
            
        Returns:
            tuple: (pan_angle, tilt_angle)
        """
        # First, ensure x and y are within valid image bounds
        x = max(0, min(self.image_width, x))
        y = max(0, min(self.image_height, y))
        
        # Normalize coordinates to [0, 1] range
        x_norm = x / self.image_width
        y_norm = y / self.image_height
        
        # Bilinear interpolation for pan angle
        # First, interpolate along the top edge (between top_left and top_right)
        pan_top = CAMERA_CORNERS['top_left']['pan'] + x_norm * (CAMERA_CORNERS['top_right']['pan'] - CAMERA_CORNERS['top_left']['pan'])
        
        # Then, interpolate along the bottom edge (between bottom_left and bottom_right)
        pan_bottom = CAMERA_CORNERS['bottom_left']['pan'] + x_norm * (CAMERA_CORNERS['bottom_right']['pan'] - CAMERA_CORNERS['bottom_left']['pan'])
        
        # Finally, interpolate between those two values based on y coordinate
        pan_angle = pan_top + y_norm * (pan_bottom - pan_top)
        
        # Bilinear interpolation for tilt angle
        # First, interpolate along the left edge (between top_left and bottom_left)
        tilt_left = CAMERA_CORNERS['top_left']['tilt'] + y_norm * (CAMERA_CORNERS['bottom_left']['tilt'] - CAMERA_CORNERS['top_left']['tilt'])
        
        # Then, interpolate along the right edge (between top_right and bottom_right)
        tilt_right = CAMERA_CORNERS['top_right']['tilt'] + y_norm * (CAMERA_CORNERS['bottom_right']['tilt'] - CAMERA_CORNERS['top_right']['tilt'])
        
        # Finally, interpolate between those two values based on x coordinate
        tilt_angle = tilt_left + x_norm * (tilt_right - tilt_left)
        
        # Ensure angles are within valid ranges
        pan_angle = max(self.pan_min, min(self.pan_max, pan_angle))
        tilt_angle = max(self.tilt_min, min(self.tilt_max, tilt_angle))
        
        # Log the mapping information for debugging
        logger.debug(f"Mapped image ({x},{y}) -> normalized ({x_norm:.2f}, {y_norm:.2f})")
        logger.debug(f"Mapped to servo angles (pan={pan_angle:.1f}, tilt={tilt_angle:.1f})")
        
        return (pan_angle, tilt_angle)
        
    def track_detection(self, detection_data):
        """
        Update the tracking target based on detection data
        
        Args:
            detection_data: Dictionary containing bounding box information
                Expected format: {'x': int, 'y': int, 'width': int, 'height': int}
                May include pre-calculated angles: {'pan_angle': float, 'tilt_angle': float}
        """
        if not detection_data:
            logger.info("No detection provided, ignoring")
            return
        
        # Store the detection data
        self.last_detection = detection_data
        
        # Check if pre-calculated angles are provided
        if 'pan_angle' in detection_data and 'tilt_angle' in detection_data:
            # Use the pre-calculated angles from the calibrated mapping
            pan_angle = detection_data['pan_angle']
            tilt_angle = detection_data['tilt_angle']
            logger.info(f"Using pre-calculated angles: pan={pan_angle:.1f}, tilt={tilt_angle:.1f}")
        else:
            # Calculate center point of the bounding box
            center_x = detection_data['x'] + (detection_data['width'] // 2)
            center_y = detection_data['y'] + (detection_data['height'] // 2)
            
            # Log the center coordinates for testing
            logger.info(f"Target center: ({center_x}, {center_y})")
            
            # Map to servo angles using the basic mapping function
            pan_angle, tilt_angle = self.map_coordinates_to_angles(center_x, center_y)
            logger.info(f"Calculated servo angles: pan={pan_angle:.1f}, tilt={tilt_angle:.1f}")
        
        # Ensure angles are within valid range
        pan_angle = max(self.pan_min, min(self.pan_max, pan_angle))
        tilt_angle = max(self.tilt_min, min(self.tilt_max, tilt_angle))
        
        # Move servos to target
        logger.info(f"Moving to servo angles: pan={pan_angle:.1f}, tilt={tilt_angle:.1f}")
        self.move(pan_angle, tilt_angle)
        
    def start_tracking(self, detection_queue=None, poll_interval=0.5):
        """Start a background thread to track detections from a queue"""
        if self.tracking_active:
            logger.warning("Tracking is already active")
            return
            
        self.tracking_active = True
        self.detection_queue = detection_queue
        
        def tracking_thread():
            logger.info("Servo tracking thread started")
            while self.tracking_active:
                if self.detection_queue and not self.detection_queue.empty():
                    # Get the next detection and process it
                    try:
                        detection = self.detection_queue.get_nowait()
                        self.track_detection(detection)
                        self.detection_queue.task_done()
                    except Exception as e:
                        logger.error(f"Error processing detection: {e}")
                
                # Brief delay to avoid hogging the CPU
                time.sleep(poll_interval)
                
            logger.info("Servo tracking thread stopped")
        
        self._tracking_thread = threading.Thread(target=tracking_thread, daemon=True)
        self._tracking_thread.start()
        logger.info("Servo tracking thread started")
        
    def stop_tracking(self):
        """Stop the continuous tracking thread"""
        if not self.tracking_active:
            return
            
        logger.info("Stopping tracking")
        self.tracking_active = False
        
        if self._tracking_thread:
            self._tracking_thread.join(timeout=2.0)
            self._tracking_thread = None
    
    def scan_corners(self, pause_time=1.0):
        """Scan the four corners of the field of view"""
        corners = [
            (CAMERA_CORNERS['top_left']['pan'], CAMERA_CORNERS['top_left']['tilt']),
            (CAMERA_CORNERS['top_right']['pan'], CAMERA_CORNERS['top_right']['tilt']),
            (CAMERA_CORNERS['bottom_right']['pan'], CAMERA_CORNERS['bottom_right']['tilt']),
            (CAMERA_CORNERS['bottom_left']['pan'], CAMERA_CORNERS['bottom_left']['tilt']),
        ]
        
        logger.info("Scanning corners")
        for i, (pan, tilt) in enumerate(corners):
            logger.info(f"Moving to corner {i+1}: pan={pan}, tilt={tilt}")
            self.move(pan, tilt)
            time.sleep(pause_time)
            
        # Return to center
        self.center()

    def cleanup(self):
        """Clean up resources"""
        self.stop_tracking()
        self.stop_pwm()
        self.pi.stop()


# Example usage (for testing)
if __name__ == "__main__":
    controller = DualServoController(
        pan_pin=DEFAULT_GPIO_PINS['pan_pin'],
        tilt_pin=DEFAULT_GPIO_PINS['tilt_pin']
    )
    
    try:
        # Center servos to start
        controller.center()
        time.sleep(1)
        
        # Test corner scanning
        controller.scan_corners(pause_time=1.5)
        time.sleep(1)
        
        # Test tracking with simulated detections
        print("\nSimulating cat detections:")
        test_detections = [
            {'x': 100, 'y': 100, 'width': 200, 'height': 150, 'confidence': 0.8},
            {'x': 400, 'y': 300, 'width': 180, 'height': 160, 'confidence': 0.9},
            {'x': 600, 'y': 200, 'width': 150, 'height': 140, 'confidence': 0.7},
            {'x': 300, 'y': 400, 'width': 220, 'height': 180, 'confidence': 0.85},
        ]
        
        for i, detection in enumerate(test_detections):
            print(f"\nDetection {i+1}:")
            print(f"Box: ({detection['x']}, {detection['y']}, {detection['width']}x{detection['height']})")
            center_x = detection['x'] + (detection['width'] // 2) 
            center_y = detection['y'] + (detection['height'] // 2)
            print(f"Center point: ({center_x}, {center_y})")
            
            controller.track_detection(detection)
            time.sleep(2)
        
        # Return to center
        controller.center()
        
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        controller.cleanup()
