
import cv2
import mediapipe as mp
import math

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )

def fingers_up(hand):
    landmarks = hand.landmark

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    thumb_up = landmarks[4].y < landmarks[3].y

    if all([index_up, middle_up, ring_up, pinky_up]):
        if thumb_up:
            return "OPEN HAND - STOP"

    if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
        return "THUMBS UP"

    if distance(landmarks[4], landmarks[8]) < 0.06:
        return "OK SIGN"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "HELP / TWO FINGERS"

    return "Gesture not recognized"

print("Gesture recognition started. Press Q to close.")

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            print("Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "Show your hand"

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                gesture = fingers_up(hand)

        cv2.putText(
            frame, gesture, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 0), 2
        )

        cv2.imshow("Healthcare Monitoring - Gesture Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
