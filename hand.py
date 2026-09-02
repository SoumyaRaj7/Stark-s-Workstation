import cv2
import mediapipe as mp

# Class responsible for detecting and tracking the hand
class HandDetector:
    def __init__(self):
        # Access the required parts of MediaPipe's Tasks API
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),
            running_mode=self.RunningMode.VIDEO,
            # Detect only one hand
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.HandLandmarker.create_from_options(options)
        self.timestamp = 0
        self.results = None

    def findHands(self, img):

        # OpenCV uses BGR images, while MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV image into a MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        self.timestamp += 1

        # Run hand detection on the current frame
        self.results = self.hands.detect_for_video(
            mp_image,
            self.timestamp
        )

    def getLandmarks(self):
        # Check whether a hand and its landmarks were detected
        if self.results and self.results.hand_landmarks:
            return self.results.hand_landmarks[0]

        return None

# Checks whether a particular finger is extended by comparing the y-coordinate of its tip and PIP joint
def finger_up(hand, tip, pip):
    return hand[tip].y < hand[pip].y

# Determines which gesture is being performed
def classify_gesture(hand):
    #Check whether each finger is extended
    index_up = finger_up(hand, 8, 6)
    middle_up = finger_up(hand, 12, 10)
    ring_up = finger_up(hand, 16, 14)
    little_up = finger_up(hand, 20, 18)

    fingers = [index_up, middle_up, ring_up, little_up]

    # Open Palm
    if all(fingers):
        return "Open Palm"

    # Closed Fist
    if not any(fingers):
        return "Closed Fist"

    # Victory
    if index_up and middle_up and not ring_up and not little_up:
        return "Victory"

    # Pointing
    if index_up and not middle_up and not ring_up and not little_up:
        return "Pointing"

    return "Unknown"

# Draws the detected hand landmarks and their connections
def draw_landmarks(img, hand):
    h, w, _ = img.shape

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    for landmark in hand:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)

    for start, end in connections:
        x1 = int(hand[start].x * w)
        y1 = int(hand[start].y * h)

        x2 = int(hand[end].x * w)
        y2 = int(hand[end].y * h)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        if not success:
            break

        detector.findHands(img)
        hand = detector.getLandmarks()

        if hand is not None:
            draw_landmarks(img, hand)
            gesture = classify_gesture(hand)
        else:
            gesture = "No Hand Detected"

        # Display the detected gesture on the webcam feed
        cv2.putText(
            img,
            gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        cv2.imshow("Hand Gesture Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()