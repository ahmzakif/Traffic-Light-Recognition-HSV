import cv2
import numpy as np


def hsv_track_tune(source, *colors):
    """
    Create HSV trackbars for specified colors (r: red, g: green, y: yellow).
    Supports video files, live camera feed, or images (jpg/png).

    Parameters:
        source: str or int
            Path to video/image file, or webcam index (e.g., 0 for the default camera).
        colors: Tuple of color initials to enable trackbars (e.g., r, g, y).
    """
    # Open video/image or webcam source
    is_image = isinstance(source, str) and (source.endswith('.jpg') or source.endswith('.png'))
    cap = cv2.VideoCapture(source) if not is_image else None

    def nothing(x):
        pass

    # Define color windows and trackbars
    trackbars = {
        'r': "Red Trackbars",
        'g': "Green Trackbars",
        'y': "Yellow Trackbars"
    }

    for color in colors:
        if color in trackbars:
            cv2.namedWindow(trackbars[color])
            cv2.createTrackbar("L - H", trackbars[color], 0, 179, nothing)
            cv2.createTrackbar("L - S", trackbars[color], 0, 255, nothing)
            cv2.createTrackbar("L - V", trackbars[color], 0, 255, nothing)
            cv2.createTrackbar("U - H", trackbars[color], 179, 179, nothing)
            cv2.createTrackbar("U - S", trackbars[color], 255, 255, nothing)
            cv2.createTrackbar("U - V", trackbars[color], 255, 255, nothing)

    while True:
        if is_image:
            frame = cv2.imread(source)
        else:
            ret, frame = cap.read()
            if not ret:
                # If video reaches the end, rewind to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # Resize frame to half its original size
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = {}
        results = {}

        for color in colors:
            if color in trackbars:
                l_h = cv2.getTrackbarPos("L - H", trackbars[color])
                l_s = cv2.getTrackbarPos("L - S", trackbars[color])
                l_v = cv2.getTrackbarPos("L - V", trackbars[color])
                u_h = cv2.getTrackbarPos("U - H", trackbars[color])
                u_s = cv2.getTrackbarPos("U - S", trackbars[color])
                u_v = cv2.getTrackbarPos("U - V", trackbars[color])

                lower = np.array([l_h, l_s, l_v])
                upper = np.array([u_h, u_s, u_v])

                mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask=mask)

                masks[color] = mask
                results[color] = result

        # Combine masks for visualization
        combined_mask = None
        for mask in masks.values():
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

        combined_result = cv2.bitwise_and(frame, frame, mask=combined_mask) if combined_mask is not None else frame

        # Display results
        cv2.imshow("Original Frame", frame)
        if 'r' in masks:
            cv2.imshow("Red Mask", masks['r'])
            cv2.imshow("Red Result", results['r'])
        if 'y' in masks:
            cv2.imshow("Yellow Mask", masks['y'])
            cv2.imshow("Yellow Result", results['y'])
        if 'g' in masks:
            cv2.imshow("Green Mask", masks['g'])
            cv2.imshow("Green Result", results['g'])

        cv2.imshow("Combined Result", combined_result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()



# hsv_track_tune('data/videos/input_video.mp4', 'r', 'g', 'y')
hsv_track_tune('data/images/redyellow2.jpg', 'r', 'y')
