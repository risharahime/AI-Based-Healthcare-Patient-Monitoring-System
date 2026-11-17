# AI-Based Healthcare Patient Monitoring System Using Emotion and Hand Gesture Recognition

## Project Overview

This project is a Python-based healthcare monitoring prototype that combines facial emotion recognition, facial landmark detection, eye direction estimation, and hand gesture recognition using a webcam.

It is designed as a demonstration of how computer vision and AI techniques can support patient monitoring and communication in healthcare environments.

## Objectives

- Detect facial emotions in real time.
- Identify facial landmarks and estimate eye direction.
- Recognize selected hand gestures.
- Display alerts when specific gestures or distress-like expressions are detected.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- DeepFace
- NumPy

## Main Modules

### 1. Facial Emotion Recognition
Uses DeepFace to estimate the dominant facial emotion, such as happy, sad, angry, fear, surprise, disgust, or neutral.

### 2. Face Landmark Detection
Uses MediaPipe Face Mesh to identify facial landmarks.

### 3. Eye Direction Estimation
Uses iris landmark positions to estimate whether the person is looking left, right, or center.

### 4. Hand Gesture Recognition
Uses MediaPipe Hands to recognize selected gestures:

- Open Hand / Stop
- Thumbs Up
- OK Sign
- Two Fingers / Help

### 5. Alert System
Displays on-screen demonstration alerts when selected hand gestures or distress-like facial expressions are detected.

## Project Structure

```text
HealthcareMonitoring/
│
├── main.py
├── emotion_test.py
├── gesture_test.py
├── hand_test.py
├── webcam_test.py
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd HealthcareMonitoring
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

5. Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main application:

```bash
python main.py
```

Allow webcam access when prompted. Press `Q` to close the application.

## Important Notes

- This project is an educational prototype and is not a clinically validated healthcare system.
- Facial emotion recognition provides estimates and cannot confirm a person's actual emotional or medical condition.
- Eye direction is approximate.
- Alerts are displayed locally on the screen; the current implementation does not send notifications to healthcare staff.
- A working webcam is required.

## Future Enhancements

- Integrate real-time caregiver notification services.
- Improve gesture recognition and emergency detection.
- Explore more robust emotion classification models.
- Develop a user-friendly healthcare monitoring interface.

## Author

**Risha Rahim E**

B.Tech Computer Science and Engineering | 2026