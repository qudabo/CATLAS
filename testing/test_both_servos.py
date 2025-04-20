# Run with:
# python3 /home/mason/camerastuff/frontend/test_both_servos.py

#!/usr/bin/env python3
"""
Script to control both pan and tilt servos using the DualServoController class.
This script demonstrates how to control both servos in a sequence.
"""

import time
import sys
import atexit
import warnings
import math  # Added for sine and cosine calculations

# Monkey patch the RPi.GPIO library to prevent cleanup errors
# This needs to be done before importing any modules that might use RPi.GPIO
try:
    import RPi.GPIO as GPIO
    
    # Save the original __del__ methods
    if hasattr(GPIO, 'PWM') and hasattr(GPIO.PWM, '__del__'):
        original_pwm_del = GPIO.PWM.__del__
        
        # Replace with a safer version that catches the specific errors
        def safe_pwm_del(self):
            try:
                original_pwm_del(self)
            except TypeError:
                pass  # Ignore TypeError during cleanup
            except Exception as e:
                print(f"Warning: Unhandled exception in PWM cleanup: {e}")
                
        # Apply the monkey patch
        GPIO.PWM.__del__ = safe_pwm_del
        
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
except Exception as e:
    print(f"Warning: Could not patch RPi.GPIO: {e}")

# Now import our servo controller
from servo_controller import DualServoController

# Global controller variable to ensure proper cleanup
controller = None

def cleanup():
    """Ensure GPIO cleanup happens exactly once before exit"""
    global controller
    if controller is not None:
        try:
            print("Cleaning up GPIO pins")
            controller.cleanup()
            controller = None
        except Exception as e:
            # Suppress any exceptions during cleanup
            pass
            
# Register cleanup to happen at exit
atexit.register(cleanup)

def main():
    global controller
    # GPIO pins for servos
    PAN_GPIO_PIN = 18
    TILT_GPIO_PIN = 17  

    # Camera corner positions in servo angles
    TOP_LEFT_PAN = 135
    TOP_LEFT_TILT = 120

    TOP_RIGHT_PAN = 30
    TOP_RIGHT_TILT = 120

    BOTTOM_LEFT_PAN = 135
    BOTTOM_LEFT_TILT = 70

    BOTTOM_RIGHT_PAN = 30
    BOTTOM_RIGHT_TILT = 75

    
    # Initialize both pan and tilt servo controller
    print(f"Initializing servos - Pan: GPIO {PAN_GPIO_PIN}, Tilt: GPIO {TILT_GPIO_PIN}")
    try:
        controller = DualServoController(pan_pin=PAN_GPIO_PIN, tilt_pin=TILT_GPIO_PIN)
        
        # Move both servos to center position (90 degrees)
        print("Moving both servos to camera center")
        controller.move_pan(135)
        controller.move_tilt(120)
        time.sleep(1)

        controller.move_pan(135)
        controller.move_tilt(120)
        time.sleep(1)

        controller.move_pan(125)
        controller.move_tilt(110)
        time.sleep(1)

        
        
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    # Explicitly call cleanup before exiting
    cleanup()
    print("Done!")
    
    # Disable system exception hooks during final Python shutdown
    # to prevent error messages from being displayed
    sys.excepthook = lambda *args: None