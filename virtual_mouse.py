import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Directions:
# Move your index finger to control the cursor.
# Quickly pinch your thumb and index finger to left-click.
# Hold the thumb-index pinch to drag.
# Pinch your thumb and middle finger to right-click.


# Handles MediaPipe hand landmark detection for webcam frames
class HandDetector:
    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.RunningMode = mp.tasks.vision.RunningMode

        # Configure the hand landmark model for continuous video input
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),
            running_mode=self.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.HandLandmarker.create_from_options(options)
        self.timestamp = 0
        self.results = None

    def find_hand(self, img):
        # Convert the OpenCV frame from BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        # Each frame requires an increasing timestamp in VIDEO mode
        self.timestamp += 1

        self.results = self.hands.detect_for_video(
            mp_image,
            self.timestamp
        )

        # Return the first detected hand, if present
        if self.results.hand_landmarks:
            return self.results.hand_landmarks[0]

        return None


# Distance between two normalized landmark positions
def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def main():

    # Initialize the webcam and hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    # Create a resizable display window
    cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Virtual Mouse", 1280, 960)

    # Obtain the current screen dimensions for coordinate mapping
    screen_width, screen_height = pyautogui.size()

    # Exponential smoothing factor for reducing pointer jitter
    smoothing = 0.25

    previous_x = screen_width / 2
    previous_y = screen_height / 2

    # Cooldown prevents a single gesture from generating repeated clicks
    click_cooldown = 0.4
    last_right_click = 0
    last_left_click = 0

    # State variables used to distinguish clicking from dragging
    left_was_pinched = False
    pinch_start_time = None
    dragging = False

    while True:

        # Capture the next webcam frame
        success, img = cap.read()

        if not success:
            break

        # Mirror the webcam feed for intuitive hand movement
        img = cv2.flip(img, 1)

        # Detect the hand and retrieve its landmarks
        hand = detector.find_hand(img)

        if hand is not None:

            # ---------------------------------------
            # POINTER CONTROL
            # ---------------------------------------

            # Landmark 8 corresponds to the index fingertip
            index_tip = hand[8]

            # Convert normalized landmark coordinates into image coordinates
            camera_x = index_tip.x * img.shape[1]
            camera_y = index_tip.y * img.shape[0]

            # Map camera coordinates to the full screen
            screen_x = (
                camera_x / img.shape[1]
            ) * screen_width

            screen_y = (
                camera_y / img.shape[0]
            ) * screen_height

            # Apply exponential smoothing to reduce small tracking variations
            current_x = previous_x + (
                screen_x - previous_x
            ) * smoothing

            current_y = previous_y + (
                screen_y - previous_y
            ) * smoothing

            # Move the system pointer to the smoothed position
            pyautogui.moveTo(
                current_x,
                current_y,
                _pause=False
            )

            previous_x = current_x
            previous_y = current_y

            # ---------------------------------------
            # GESTURE DETECTION
            # ---------------------------------------

            # Relevant fingertip landmarks for the gesture controls
            thumb_tip = hand[4]
            index_tip = hand[8]
            middle_tip = hand[12]

            # Use palm width as a reference to make pinch detection
            # less dependent on the hand's distance from the camera
            palm_size = distance(hand[5], hand[17])

            # Measure the two pinch distances
            thumb_index_distance = distance(
                thumb_tip,
                index_tip
            )

            thumb_middle_distance = distance(
                thumb_tip,
                middle_tip
            )

            # A pinch is detected when the fingertips are sufficiently close
            left_pinch = (
                thumb_index_distance < palm_size * 0.35
            )

            right_pinch = (
                thumb_middle_distance < palm_size * 0.35
            )

            # ---------------------------------------
            # LEFT CLICK / DRAG
            # ---------------------------------------

            if left_pinch:

                # Record the beginning of a new pinch
                if not left_was_pinched:
                    left_was_pinched = True
                    pinch_start_time = time.time()

                # A pinch held beyond the threshold initiates dragging
                if (
                    not dragging
                    and time.time() - pinch_start_time >= 0.25
                ):
                    pyautogui.mouseDown()
                    dragging = True

            else:

                # Process the pinch when the fingers separate
                if left_was_pinched:

                    pinch_duration = (
                        time.time() - pinch_start_time
                    )

                    # A short pinch is interpreted as a left click
                    if not dragging and pinch_duration < 0.25:

                        if (
                            time.time() - last_left_click
                            > click_cooldown
                        ):
                            pyautogui.click()
                            last_left_click = time.time()

                    # Releasing a drag releases the mouse button
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                    left_was_pinched = False
                    pinch_start_time = None

            # ---------------------------------------
            # RIGHT CLICK
            # ---------------------------------------

            # Thumb-middle pinch acts as the right-click gesture
            if right_pinch and not left_pinch:

                if (
                    time.time() - last_right_click
                    > click_cooldown
                ):
                    pyautogui.rightClick()
                    last_right_click = time.time()

        else:

            # With no hand detected, the pointer remains stationary

            # Release an active drag if the hand leaves the frame
            if dragging:
                pyautogui.mouseUp()
                dragging = False

            left_was_pinched = False
            pinch_start_time = None

        # Display the webcam feed
        cv2.imshow("Virtual Mouse", img)

        # Press Q to terminate the program
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Ensure the mouse button is released before exiting
    if dragging:
        pyautogui.mouseUp()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()