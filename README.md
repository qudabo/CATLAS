# CATLAS - Cat Automated Tracking Laser System

CATLAS is an intelligent laser pointer system that automatically detects and tracks cats using computer vision. The system uses a Raspberry Pi, servo motors, and a camera to create an interactive laser toy that can operate both autonomously and through manual control.

## Features

- **Real-time Cat Detection**: Uses YOLOv4-tiny for efficient cat detection
- **Autonomous Tracking**: Automatically follows detected cats with the laser pointer
- **Manual Control**: Web interface for direct control of the laser pointer
- **Safety Features**: 
  - Aims slightly below detected cats for eye safety
  - Automatic laser shutoff after periods of inactivity
  - Configurable movement limits and speeds

## System Components

- **Frontend**: Web interface and camera streaming server (Flask)
- **Backend**: Cat detection service using YOLO
- **Hardware**:
  - Raspberry Pi 4
  - Two servo motors (pan/tilt)
  - Laser module
  - Camera module
  - GPIO pins for hardware control

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Laser-Turret
```

2. Install dependencies:
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate laser-turret
```

## Configuration

Key configuration files are located in the `shared` directory:
- `servo_config.py`: Default settings for servo movements and GPIO pins
- `models/yolo/`: YOLO model configuration and weights

### Default GPIO Pin Assignments:
- Pan Servo: GPIO 13 (Hardware PWM)
- Tilt Servo: GPIO 12 (Hardware PWM)
- Laser Control: GPIO 27

## Usage

### Starting the System

1. Start the backend detection service:
```bash
python backend/backend.py
```

2. Start the frontend server:
```bash
python frontend/frontend.py
```

3. Access the web interface:
- Main interface: http://[raspberry-pi-ip]:5000/CATLAS
- Remote control: http://[raspberry-pi-ip]:5000/remote_control

### Web Interface Features

- Live video feed with cat detection overlay
- Manual control zone for direct laser positioning
- Tracking controls (Start/Stop)
- Servo controls (Center/Test Corners)
- Laser controls (On/Off)
- System status monitoring
- Activity log

## Testing

Various test scripts are provided in the `testing` directory:
- `test_both_servos.py`: Verify servo operation
- `cat_laser.py`: Test the complete tracking system
- `track_cat.py`: Test cat detection and tracking

## Safety Notes

1. The system includes safety features to avoid direct eye contact:
   - Aims below detected cats
   - Includes movement limits
   - Has auto-shutoff functionality

2. Manual supervision is recommended during operation

## Project Structure

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

## Advanced Configuration

Key settings can be modified in `shared/servo_config.py`:
- Movement speeds and delays
- Servo angle limits
- Camera resolution
- Safety timeouts

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Insert License Information]
