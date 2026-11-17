
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam is working! Press Q to close the camera window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam.")
            break

        cv2.imshow("Healthcare Monitoring - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
