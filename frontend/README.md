# Laser Turret Frontend

This directory contains the frontend application that runs on the Raspberry Pi 4 to capture images from the camera and communicate with the backend for cat detection.

Testing without servos:
python frontend/test_tracking.py --backend-url http://YOUR_BACKEND_IP:5001


Testing with servos:
python frontend/track_cat.py --backend-url http://YOUR_BACKEND_IP:5001 --pan-pin 17 --tilt-pin 27