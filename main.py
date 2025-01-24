import torch

from autoencoder import Decoder, Encoder
from device import device
from settings import *
from stepper import Stepper
from utils import load_video, make_or_load_model, video_display

# Initialize stepper

stepper = make_or_load_model(Stepper, 'stepper', Encoder(), Decoder()).to(device)

# Load video

frames = load_video(video_path, start=0, num_frames=10, num_frames_per_frame=20)

# Generate frames

frames = stepper.generate(frames, num_frames=5)

video_display(frames)