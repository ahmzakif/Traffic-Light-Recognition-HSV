# Traffic Light Recognition Using HSV

This project provides a Python-based solution to detect traffic light signals in videos using HSV color space. The implementation uses OpenCV for image processing and provides functionalities such as low-light enhancement, ROI (Region of Interest) selection, and logging detected signals with timestamps.

## Features

- **Traffic Light Detection:** Detects red, yellow, and green lights using HSV color ranges.
- **Low-Light Handling:** Enhances video frames using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve detection in low-light conditions.
- **Interactive ROI Selection:** Allows the user to define a specific area in the frame for detection, reducing false positives.
- **Logging:** Logs detected traffic light signals along with their positions and timestamps.
- **FPS Display:** Displays the current frames per second (FPS) on the video.

## Usage

1. **Input Video:** Place your input video in the `data/videos/` directory.

2. **Run the Script:**
   ```bash
   python traffic_light_detector.py
   ```

3. **Interactive ROI Selection:** The program will prompt a window to select a region of interest (ROI). Use your mouse to draw a rectangle around the area of interest and press Enter.

4. **Output Video:** Processed video with detected traffic lights will be saved in the `data/videos/` directory with the name `output_video1.mp4`.

5. **Logs:** Detection logs are saved in `data/log/traffic_light_log_1.txt`.

## Example Output

Here are examples of the processed video outputs:

### Output Video 1
![Output Video 1](data/videos/output_video1.mp4)

### Output Video 2
![Output Video 2](data/videos/output_video2.mp4)

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

## Limitations

- Works best for traffic lights with clear visibility and less occlusion.
- ROI selection is manual and may need to be adjusted for different videos.

