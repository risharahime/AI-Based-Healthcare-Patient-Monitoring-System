
import cv2
import mediapipe as mp
from deepface import DeepFace
import time
import math

# 1. Initialize webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

# 2. Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Face landmark connections
face_connections = mp_face_mesh.FACEMESH_TESSELATION

emotion = "Detecting..."
gesture = "No hand detected"
eye_direction = "Detecting..."
last_analysis_time = 0
alert_message = "Monitoring started"


def get_distance(point1, point2):
    return math.hypot(
        point1.x - point2.x,
        point1.y - point2.y
    )


def recognize_gesture(hand):
    points = hand.landmark

    index_up = points[8].y < points[6].y
    middle_up = points[12].y < points[10].y
    ring_up = points[16].y < points[14].y
    pinky_up = points[20].y < points[18].y
    thumb_up = points[4].y < points[3].y

    if all([index_up, middle_up, ring_up, pinky_up]):
        return "STOP / OPEN HAND"

    if thumb_up and not any([
        index_up, middle_up, ring_up, pinky_up
    ]):
        return "THUMBS UP"

    if get_distance(points[4], points[8]) < 0.06:
        return "OK SIGN"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "HELP / TWO FINGERS"

    return "Unrecognized gesture"


def estimate_eye_direction(face_landmarks):
    # Approximate eye direction using iris positions
    points = face_landmarks.landmark

    # Left and right eye corners
    left_outer = points[33]
    left_inner = points[133]
    right_inner = points[362]
    right_outer = points[263]

    # Iris center landmarks in MediaPipe Face Mesh
    left_iris = points[468]
    right_iris = points[473]

    left_ratio = (
        (left_iris.x - left_outer.x) /
        max(abs(left_inner.x - left_outer.x), 0.0001)
    )

    right_ratio = (
        (right_iris.x - right_inner.x) /
        max(abs(right_outer.x - right_inner.x), 0.0001)
    )

    average_ratio = (left_ratio + right_ratio) / 2

    if average_ratio < 0.40:
        return "Looking LEFT (estimate)"
    elif average_ratio > 0.60:
        return "Looking RIGHT (estimate)"
    else:
        return "Looking CENTER (estimate)"


print("Healthcare monitoring started. Press Q to exit.")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands, mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read webcam.")
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()

        # 3. Estimate facial emotion every 2 seconds
        if current_time - last_analysis_time >= 2:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True
                )
                emotion = result[0]["dominant_emotion"]
            except Exception as error:
                print("Emotion analysis error:", error)

            last_analysis_time = current_time

        # 4. Detect facial landmarks and estimate eye direction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        eye_direction = "Face not detected"

        if face_results.multi_face_landmarks:
            for face in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face,
                    connections=face_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 180, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )

                eye_direction = estimate_eye_direction(face)

        # 5. Detect hand landmarks
        hand_results = hands.process(rgb_frame)
        gesture = "No hand detected"

        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                gesture = recognize_gesture(hand)

        # 6. Demonstration alert conditions
        alert_message = "Monitoring normally"

        if gesture in [
            "STOP / OPEN HAND",
            "HELP / TWO FINGERS"
        ]:
            alert_message = "CAREGIVER ALERT: Gesture detected"
        elif emotion in ["angry", "fear", "sad"]:
            alert_message = "NOTICE: Distress-like expression estimated"

        # 7. Display results
        cv2.putText(
            frame, "Emotion: " + emotion,
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2
        )

        cv2.putText(
            frame, "Gesture: " + gesture,
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2
        )

        cv2.putText(
            frame, "Eyes: " + eye_direction,
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 180, 0), 2
        )

        cv2.putText(
            frame, alert_message,
            (20, 125), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2
        )

        cv2.imshow("AI Healthcare Patient Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("Monitoring stopped.")
