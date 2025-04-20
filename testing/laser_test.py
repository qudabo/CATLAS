#!/usr/bin/env python3
"""
Laser Control Test Script for 2N3906 PNP Transistor

This script tests the laser control functionality using a 2N3906 PNP transistor
as a high-side switch connected to GPIO pin 27.

How it works:
- With a PNP transistor as high-side switch, logic is inverted:
  - GPIO HIGH turns the laser OFF (transistor not conducting)
  - GPIO LOW turns the laser ON (transistor conducting)

Usage:
    python laser_test.py [options]

Options:
    --blink N      Number of times to blink the laser (default: 5)
    --delay SEC    Delay between on/off cycles in seconds (default: 1.0)
    --pin PIN      GPIO pin number for laser control (default: 27)
"""

import RPi.GPIO as GPIO
import time
import argparse
import logging
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Default settings
LASER_PIN = 27  # GPIO pin 27 for laser control

class LaserController:
    """Class to control the laser module with 2N3906 PNP transistor"""
    
    def __init__(self, pin=LASER_PIN):
        """Initialize the laser controller"""
        self.pin = pin
        self.is_on = False
        
        # Reset any existing GPIO settings to avoid conflicts
        try:
            GPIO.cleanup()
        except:
            pass
        
        # Setup GPIO with BCM mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)  # Disable warnings that might confuse users
        
        # Setup laser pin as output (HIGH = laser OFF, LOW = laser ON)
        # For PNP transistor, HIGH turns laser OFF (initial state)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.HIGH)
        logger.info(f"Laser controller initialized with pin {self.pin}")
        logger.info("Using 2N3906 PNP transistor as high-side switch")
    
    def turn_on(self):
        """Turn the laser on"""
        # For PNP transistor: Drive pin LOW to turn laser ON
        GPIO.output(self.pin, GPIO.LOW)
        self.is_on = True
        logger.info("Laser turned ON (GPIO set to LOW)")
    
    def turn_off(self):
        """Turn the laser off"""
        # For PNP transistor: Drive pin HIGH to turn laser OFF
        GPIO.output(self.pin, GPIO.HIGH)
        self.is_on = False
        logger.info("Laser turned OFF (GPIO set to HIGH)")
    
    def toggle(self):
        """Toggle the laser state"""
        if self.is_on:
            self.turn_off()
        else:
            self.turn_on()
    
    def blink(self, count=5, delay=1.0):
        """Blink the laser on and off for a specified number of times"""
        logger.info(f"Blinking laser {count} times with {delay}s delay")
        
        for i in range(count):
            logger.info(f"Blink {i+1}/{count}")
            self.turn_on()
            time.sleep(delay)
            self.turn_off()
            time.sleep(delay)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        # Make sure laser is off before cleanup
        if self.is_on:
            self.turn_off()
        
        GPIO.cleanup()
        logger.info("GPIO resources cleaned up")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Laser Control Test Script")
    
    parser.add_argument("--blink", 
                        type=int,
                        default=5, 
                        help="Number of times to blink the laser")
    parser.add_argument("--delay",
                        type=float,
                        default=1.0,
                        help="Delay between on/off cycles in seconds")
    parser.add_argument("--pin",
                        type=int,
                        default=LASER_PIN,
                        help="GPIO pin number for laser control")
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize laser controller
    laser = LaserController(pin=args.pin)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Signal received, cleaning up...")
        laser.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Test basic on/off functionality first
        logger.info("Testing laser ON/OFF...")
        laser.turn_on()
        time.sleep(1)
        laser.turn_off()
        time.sleep(1)
        
        # Blink the laser
        logger.info(f"Starting blink test: {args.blink} cycles with {args.delay}s delay")
        laser.blink(count=args.blink, delay=args.delay)
        
        # Pattern test (SOS - ... --- ...)
        logger.info("Performing SOS pattern test")
        
        # S - three short pulses
        for _ in range(3):
            laser.turn_on()
            time.sleep(0.2)
            laser.turn_off()
            time.sleep(0.2)
            
        time.sleep(0.5)  # Pause between letters
        
        # O - three long pulses
        for _ in range(3):
            laser.turn_on()
            time.sleep(0.6)
            laser.turn_off()
            time.sleep(0.2)
            
        time.sleep(0.5)  # Pause between letters
        
        # S - three short pulses again
        for _ in range(3):
            laser.turn_on()
            time.sleep(0.2)
            laser.turn_off()
            time.sleep(0.2)
        
        logger.info("Test completed. Laser is OFF.")
        
    finally:
        # Clean up GPIO resources
        laser.cleanup()


if __name__ == "__main__":
    main()