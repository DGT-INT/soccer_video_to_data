# soccer_video_to_data
Smart Soccer Insights uses computer vision to convert soccer game footage into structured coordinate and event data. With deep learning for detection, tracking, and action recognition, it enables downstream analysis of player movement, tactics, and performance.

# Single-Camera Soccer Player Tracking

This project extracts player coordinates from standard broadcast soccer footage using a single-camera computer vision pipeline.

## Motivation

Professional clubs often rely on expensive multi-camera systems or GPS trackers to analyze player movement.
Lower-budget teams typically lack access to these tools.

This project explores whether reliable player tracking can be achieved using standard broadcast footage and open-source computer vision techniques.

## Repository Overview

- `app/` – Streamlit application for running the tracking pipeline
- `src/` – Core computer vision and tracking logic
- `paper/` – Research paper describing the methodology and evaluation
- `data/` – Sample input footage and outputs

## Demo
Demo coming soon!!!

## Method Overview

1. Input broadcast soccer footage
2. Detect players using object detection
3. Track players across frames
4. Transform image coordinates into a normalized pitch representation
5. Output tracked player trajectories and annotated video

## Streamlit App

The app allows users to:
- Upload custom footage or use a sample video
- Run the player detection and tracking pipeline
- View the processed video output

The app includes loading indicators and caching to handle long-running computations.

## Paper

A detailed explanation of the methodology, experiments, and evaluation can be found in the accompanying paper:

paper coming soon!!!
📄 `paper/your_paper_name.pdf`

## Future Work

TBD

## Status
This project is currently in progress. Features, models, and documentation are actively being developed.



