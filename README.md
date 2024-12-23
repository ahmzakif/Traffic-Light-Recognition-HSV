# Traffic Light Recognition Using HSV

This project provides a Python-based solution to detect traffic light signals in videos using HSV color space. The implementation uses OpenCV for image processing and provides functionalities such as low-light enhancement, ROI (Region of Interest) selection, and logging detected signals with timestamps.

## Features

- **Traffic Light Detection:** Detects red, yellow, and green lights using HSV color ranges.
- **Low-Light Handling:** Enhances video frames using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve detection in low-light conditions.
- **Interactive ROI Selection:** Allows the user to define a specific area in the frame for detection, reducing false positives.
- **Logging:** Logs detected traffic light signals along with their positions and timestamps.
- **FPS Display:** Displays the current frames per second (FPS) on the video.
- **HSV Tuning Utility:** Provides an interactive tool to tune HSV color ranges for traffic light detection.

## How It Works

1. **Preprocessing:** Enhances video frames using CLAHE to handle low-light conditions.
2. **Color Detection:** Detects traffic light colors based on predefined HSV ranges:
   - **Red:** Two ranges to cover different shades.
   - **Yellow:** Single range.
   - **Green:** Single range.
3. **Interactive ROI Selection:**
   - The user selects the Region of Interest (ROI) by drawing a rectangle on the video frame.
   - Detection is limited to the selected ROI.
4. **HSV Range Tuning:**
   - Interactive tool to adjust HSV ranges for red, yellow, and green colors.
   - Supports video, image, or webcam input.
5. **Video Processing:**
   - Processes each frame, detects traffic lights within the ROI, and logs detections.
   - Displays the output in real-time with FPS overlay.
6. **Output Video:** Saves the processed video with detected traffic lights marked.

## HSV Tuning Utility

The `hsv_track_tune.py` script allows you to interactively adjust HSV ranges for red, yellow, and green colors using trackbars. This is especially useful for fine-tuning detection thresholds for different lighting conditions.

### Usage

1. **Run the script:**
   ```bash
   python hsv_track_tune.py
   ```
2. Specify the source (image/video/webcam) and the colors to tune (`r`, `g`, `y`).
3. Adjust the HSV trackbars for each color to observe the changes in real-time.
4. Press `q` to exit.

### Example

```python
hsv_track_tune('data/videos/input_video.mp4', 'r', 'g', 'y')
hsv_track_tune('data/images/input_image.jpg', 'r', 'y')
```

### Explanation

- **Source:** Accepts a path to a video, an image, or a webcam index (e.g., `0` for the default camera).
- **Colors:** Specify the colors to tune using their initials (`r` for red, `g` for green, `y` for yellow).
- **Output:** Displays the original frame, masks, and results for each selected color.

## Usage

1. **Input Video:** Place your input video in the `data/videos/` directory.
2. **Run the Script:**

   ```bash
   python traffic_light_detector.py
   ```
3. **Interactive ROI Selection:** The program will prompt a window to select a region of interest (ROI). Use your mouse to draw a rectangle around the area of interest and press Enter.
4. **Output Video:** Processed video with detected traffic lights will be saved in the `data/videos/` directory with the name `output_video.mp4`.
5. **Logs:** Detection logs are saved in `data/log/traffic_light_log.txt`.

## Example Output

Here are examples of the processed video outputs:

### Output Video 

![Output Video 1](data/videos/output1.gif)

![Output Video 2](data/videos/output2.gif)

## Key Functions

### Class: `TrafficLightDetector`

#### Methods:

- **`__init__(log_file)`**
  Initializes the detector and sets up HSV color ranges, CLAHE parameters, and logging.
- **`preprocess_frame(frame)`**
  Enhances the input frame for better visibility in low-light conditions.
- **`detect_traffic_lights(frame, mask)`**
  Detects traffic lights within the ROI using HSV masks.
- **`show_fps(frame, prev_time)`**
  Displays the current FPS on the video frame.
- **`select_roi(frame)`**
  Allows the user to select a region of interest (ROI) interactively.
- **`process_video(input_video_path, output_video_path)`**
  Processes the input video and saves the output with detected traffic lights.

## Configuration

- **Log File Path:** Set the log file path in the `__init__` method.
- **Color Ranges:** Modify the HSV color ranges for red, yellow, and green in `self.color_ranges` if needed.

## Logging

Detected traffic lights are logged in `data/log/traffic_light_log_1.txt`. Each log entry includes:

- Timestamp
- Detected color
- Position and size of the traffic light in the frame

Example log entry:

```
2024-12-16 14:32:10 - [2024-12-16 14:32:10] Detected RED light at position 120,200
```

## Limitations

- Works best for traffic lights with clear visibility and less occlusion.
- ROI selection is manual and may need to be adjusted for different videos.
