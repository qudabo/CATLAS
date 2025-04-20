**PROJECT STRUCTURE**

Laser-Turret/
├── backend/                 # Backend code for cat detection processing
│   ├── backend.py           # Main backend service
│   └── README.md            # Documentation for the backend component
├── frontend/                # Frontend code for Raspberry Pi camera interface
│   ├── frontend.py          # Main frontend application 
│   └── README.md            # Documentation for the frontend component
├── shared/                  # Code shared between frontend and backend
│   ├── cat_detector.py      # Cat detection implementation using YOLO
│   ├── README.md            # Documentation for shared components
│   └── models/              # Model files directory
│       └── yolo/            # YOLO model directory
│           ├── coco.names   # COCO dataset class names
│           ├── yolov4-tiny.cfg  # YOLO configuration file
│           └── yolov4-tiny.weights  # YOLO model weights
├── pyproject.toml           # Project Python package configuration
├── environment.yml          # Conda environment configuration
└── README.md                # Main project documentation


bilinear interpolation to map any pixel in the camera image to its corresponding pan and tilt angles.

cat laser code:
python cat_laser.py --pan-speed 2.0 --tilt-speed 2.0 --move-delay 0.2