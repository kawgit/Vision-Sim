import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

from device import device

def load_video(video_path, start=0, num_frames=100, num_frames_per_frame=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    
    frames = [frame for idx, frame in enumerate(frames) if idx % num_frames_per_frame == 0][start:start + num_frames]

    frames = np.array(frames, dtype=np.float32) / 255
    frames = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2)

    return frames

def write_video(frames, output_filename, fps = 2):

    if type(frames) == torch.Tensor:
        frames = np.array(frames.detach().cpu())

    frames = (np.array(frames) * 255).astype(np.uint8)
    frames = np.array([cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR) for frame in frames])

    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError("Input frames should be a 4D numpy array with shape (N, frame_height, frame_width, 3).")

    print(frames.shape)
    print(frames.mean(), frames.std())

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frames.shape[-2:])

    for frame in frames:
        out.write(frame)

    out.release()

def img_normalize(img):
    return img.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

def img_display(img):
    plt.imshow(img_normalize(img))

def img_save(img, path='output.png'):
    plt.imsave(path, img_normalize(img))

def get_last_file(directory_path):
    files = glob.glob(os.path.join(directory_path, '*'))
    files.sort()
    return files[-1] if files else None

def get_model_dir(model_name):

    model_dir = os.path.join("models", f"{model_name}s")

    os.makedirs(model_dir, exist_ok=True)

    return model_dir

def make_or_load_model(model_class, model_name, *args, **kwargs):

    model = model_class(*args, **kwargs)
    model_path = get_last_file(get_model_dir(model_name))

    if model_path:
        print(f"Loading weights from {model_path}.")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model

def format_number(number, total_length=7):
    return f"{number:0{total_length}.6f}"


def save_model(model, model_name, cost=0):
    torch.save(model.state_dict(), os.path.join(get_model_dir(model_name), f"{round(time.time())}_{format_number(cost)}.pt"))

def video_display(frames):

    for frame in frames:

        img_save(frame)

        time.sleep(.5)