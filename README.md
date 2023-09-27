# Vuae - Video Upscaler and Enhancer

**Developed by 4-division**


## Description

Vuae is a video upscaler developed with Python, FFmpeg, and REALesrgan.
The purpose of this program is to serve a GUI frontend without Python installed.
This program is for computers without GPUs that need to upscale videos or want to use AI processing.
It splits the video into separate frames, processes them, and combines them back into the final video.

## Features

- Change Video Resolution to any defined scale factor
- Effects including sharpening and deblurring
- AI Upscaling using REALesrgan
- CPU only for computers without dedicated GPUs
- Fast upscaling using efficient realesr-general-x4v3 pre-trained model
- GUI based

## Installation
```pip install -r requirements.txt```

```python3 Vuae.py```

## Requirements

- opencv-python
- tqdm
- tk
- numpy
- pillow
- moviepy
- glob
- platform
- realesrgan
- torch>=1.7
- torchvision
- basicsr>=1.4.2
- facexlib>=0.2.5
- gfpgan>=1.3.5

## Guide

Vuae has only been tested with mp4 files, but feel free to test other file formats. Make sure to keep video files inside the installed directory to ensure no errors. Ignore errors without AI upscaling; they are only used when AI upscaling is used.

1. Open video file in GUI.
2. Select a save destination inside the directory make sure to include the .mp4 extension.
3. Adjust parameters.
4. Click Upscale and enhance video.

## Credits

- [github.com/johnson-cooper](https://github.com/johnson-cooper)
- [github.com/aquali_xoxo](https://github.com/aquali_xoxo)
- [github.com/4-division](https://github.com/4-division)
- [4-division.com](https://4-division.com)

Thankful for these amazing projects:

- [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [github.com/Zulko/moviepy](https://github.com/Zulko/moviepy)
- [github.com/opencv/opencv](https://github.com/opencv/opencv)
