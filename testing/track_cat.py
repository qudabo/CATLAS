#!/usr/bin/env python3
"""
Cat Tracking System - Main Script

This script integrates cat detection with servo control to track cats in real-time.
The servos will automatically move to center on any detected cat.

Usage:
    python track_cat.py [--backend-url URL] [--pan-pin PIN] [--tilt-pin PIN]

Options:
    --backend-url URL   URL of the backend detection service (default: http://localhost:5001)
    --pan-pin PIN       GPIO pin for the pan servo (default: 17)  
    --tilt-pin PIN      GPIO pin for the tilt servo (default: 27)
"""

import argparse
import time
import sys
import logging
import signal

# Fix the relative import issue
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.servo_controller import DualServoController
from frontend.cat_tracker import CatTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('cat_tracking.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cat Tracking System")
    
    parser.add_argument("--backend-url", 
                        default="http://localhost:5001",
                        help="URL of the backend detection service")
    parser.add_argument("--pan-pin", 
                        type=int, 
                        default=17, 
                        help="GPIO pin for pan servo")
    parser.add_argument("--tilt-pin", 
                        type=int, 
                        default=27, 
                        help="GPIO pin for tilt servo")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting cat tracking system")
    logger.info(f"Backend URL: {args.backend_url}")
    logger.info(f"Pan servo pin: {args.pan_pin}")
    logger.info(f"Tilt servo pin: {args.tilt_pin}")
    
    # Create the cat tracker
    tracker = CatTracker(
        backend_url=args.backend_url,
        pan_pin=args.pan_pin,
        tilt_pin=args.tilt_pin
    )
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        tracker.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize by centering
        logger.info("Centering servos...")
        tracker.center()
        time.sleep(1)
        
        # Test motion with corner scan
        logger.info("Testing servo motion with corner scan...")
        tracker.test_corners()
        time.sleep(1)
        
        # Start tracking mode
        logger.info("Starting continuous cat tracking")
        logger.info("Press Ctrl+C to stop")
        tracker.start_tracking()
        
        # Keep the main thread running to allow tracking in background threads
        while True:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in tracking system: {e}")
    finally:
        # Ensure cleanup runs on exit
        tracker.cleanup()

if __name__ == "__main__":
    main()