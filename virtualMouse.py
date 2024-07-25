import mediapipe as mp
import pyautogui
import numpy as np
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def hand_gesture_mouse():
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()
    prev_x, prev_y = 0, 0  # Previous x and y coordinates of the hand
    smoothing_factor = 0.7  # Adjust the smoothing factor as needed
    click_threshold = 30  # Adjust the click threshold based on your testing

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                if lm_list:
                    x, y = lm_list[8]  # Index finger tip

                    # Apply smoothing
                    smooth_x = int(prev_x + (x - prev_x) * smoothing_factor)
                    smooth_y = int(prev_y + (y - prev_y) * smoothing_factor)

                    # Translate hand coordinates to screen coordinates
                    # Invert the horizontal movement
                    screen_x = int(screen_width / w * (w - smooth_x))
                    screen_y = int(screen_height / h * smooth_y)
                    pyautogui.moveTo(screen_x, screen_y)

                    # Update previous coordinates
                    prev_x, prev_y = smooth_x, smooth_y

                    # Calculate the distance between the index finger tip and thumb tip
                    thumb_tip_x, thumb_tip_y = lm_list[4]  # Thumb tip
                    index_tip_x, index_tip_y = lm_list[8]  # Index finger tip

                    distance = np.sqrt(
                        (thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2)

                    if distance < click_threshold:
                        pyautogui.click()

                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Gesture Mouse", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hand_gesture_mouse()
