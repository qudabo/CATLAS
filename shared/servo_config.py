#!/usr/bin/env python3
"""
Shared Servo Configuration

This file contains shared configuration settings for the camera and servo setup,
including calibrated corner positions. All modules that need these settings
should import from this file.
"""

# Calibrated camera corner positions (servo angles for each corner of the image)
CAMERA_CORNERS = {
    'top_left': {'pan': 135, 'tilt': 120},
    'top_right': {'pan': 30, 'tilt': 120},
    'bottom_left': {'pan': 135, 'tilt': 70},
    'bottom_right': {'pan': 30, 'tilt': 75},
}

# Standardized servo center position for laser
STANDARD_CENTER = {'pan': 90, 'tilt': 90}

# Servo limits
SERVO_LIMITS = {
    'pan_min': 20,  # Minimum pan angle
    'pan_max': 160, # Maximum pan angle
    'tilt_min': 55, # Minimum tilt angle
    'tilt_max': 125 # Maximum tilt angle
}

# Default GPIO pin assignments
DEFAULT_GPIO_PINS = {
    'pan_pin': 13,  # Hardware PWM pin (GPIO13)
    'tilt_pin': 12, # Hardware PWM pin (GPIO12)
    'laser_pin': 27 # GPIO pin for laser control (2N3906 PNP transistor)
}

# Default movement settings
MOVEMENT_SETTINGS = {
    'pan_step': 5.0,      # Pan speed in degrees per move
    'tilt_step': 5.0,     # Tilt speed in degrees per move
    'movement_delay': 0.1 # Delay between movements in seconds
}

# Laser control settings
LASER_SETTINGS = {
    'auto_off_delay': 5.0,  # Seconds of inactivity before auto-turnoff
    'blink_count': 3,       # Number of blinks when testing
    'blink_delay': 0.3,     # Delay between blinks in seconds
}

# Default camera settings
CAMERA_SETTINGS = {
    'width': 2304,   # Camera width in pixels
    'height': 1296,  # Camera height in pixels
    'fps': 30        # Camera frames per second
}