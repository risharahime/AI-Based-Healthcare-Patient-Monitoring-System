
import cv2
from deepface import DeepFace

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

print("Emotion detection started. Press Q to close.")

frame_count = 0
emotion = "Detecting..."

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Analyze every 15 frames to reduce processing time
    if frame_count % 15 == 0:
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

    cv2.putText(
        frame,
        "Emotion: " + emotion,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Healthcare Monitoring - Emotion Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
