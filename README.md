# Semantic YOLO Video Frame Analyzer

A Python application that combines semantic analysis (CLIP) and object detection (YOLOv5) to extract and analyze key frames from video footage.

## Overview

This project provides a sophisticated video analysis tool that identifies important key frames in videos by using two complementary approaches:

1. **Semantic Frame Analysis**: Uses OpenAI's CLIP model to calculate semantic distances between consecutive frames
2. **Object Detection Analysis**: Uses YOLOv5 to identify objects in frames and score their importance

The system combines these two approaches to identify frames that represent significant visual changes or contain important objects of interest.

## Features

- Extracts frames from videos at customizable sampling rates
- Identifies key frames using semantic distance thresholds
- Detects objects within frames using YOLOv5
- Scores frames based on object importance, with configurable priority classes
- Combines semantic and object scores with adjustable weights
- Generates detailed analysis reports and visualizations:
  - Key frames with timestamps
  - Object detection visualizations for key frames
  - Frame score graphs
  - Object class distribution charts
  - Comprehensive text report with frame details

## Installation

### Prerequisites

- Python 3.11 or higher
- PyTorch (compatible with your system)
- CUDA-capable GPU recommended for faster processing (optional)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/semantic-yolo-video-frames.git
   cd semantic-yolo-video-frames
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Usage

Basic usage:

```bash
uv run main.py --video path/to/video.mp4
```

### Command Line Arguments

| Argument             | Description                                                              | Default                |
| -------------------- | ------------------------------------------------------------------------ | ---------------------- |
| `--video`            | Path to the video file                                                   | _Required_             |
| `--output_dir`       | Directory for saving results                                             | `output`               |
| `--threshold`        | Semantic distance threshold for key frames                               | `0.2`                  |
| `--object_threshold` | Object score threshold for key frames                                    | `0.3`                  |
| `--combined_weight`  | Weight of semantic score when combining (1-weight = object score weight) | `0.5`                  |
| `--sample_rate`      | Frame sampling rate (every N-th frame)                                   | `1`                    |
| `--priority_classes` | Priority object classes, comma-separated                                 | `person,car,truck,bus` |

### Examples

Process a video with custom settings:

```bash
uv run main.py --video data/my_video.mp4 --threshold 0.3 --object_threshold 0.4 --combined_weight 0.6
```

Process a video with higher sampling rate (for long videos):

```bash
uv run main.py --video data/long_video.mp4 --sample_rate 10
```

Customize priority objects:

```bash
uv run main.py --video data/wildlife.mp4 --priority_classes "dog,cat,bird,elephant"
```

## Output

The program creates the following output structure:

```
output/
├── keyframes_info.txt        # Detailed information about all key frames
├── graphs/
│   ├── frame_scores.png      # Visual graph of all frame scores
│   └── object_classes_distribution.png  # Distribution of detected objects
├── keyframes/
│   ├── keyframe_1.jpg        # Original key frames
│   ├── keyframe_2.jpg
│   └── ...
└── yolo_frames/
    ├── yolo_keyframe_1.jpg   # Key frames with object detection visualization
    ├── yolo_keyframe_2.jpg
    └── ...
```

## How It Works

1. **Frame Extraction**: Samples frames from the input video
2. **Semantic Analysis**:
   - Generates embeddings using CLIP model
   - Calculates semantic distances between consecutive frames
3. **Object Detection**:
   - Detects objects in each frame using YOLOv5
   - Analyzes object classes, counts, and confidence levels
4. **Score Calculation**:
   - Computes semantic scores based on frame distances
   - Computes object scores based on detected objects
   - Combines scores with configured weights
5. **Key Frame Identification**:
   - Identifies frames exceeding threshold values
6. **Results Storage**:
   - Saves key frames
   - Generates visualizations
   - Creates detailed analysis report

## Technical Details

The system uses the following major components:

- **CLIP (Contrastive Language-Image Pretraining)** by OpenAI for semantic understanding
- **YOLOv5** for object detection
- **OpenCV** for video processing
- **Matplotlib** for visualization
- **PyTorch** as the deep learning framework
