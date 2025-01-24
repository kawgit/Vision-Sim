import torch

from autoencoder import Decoder, Encoder
from dataloader import build_stepper_dataloader
from device import device
from settings import *
from stepper import Stepper
from trainer import Trainer
from utils import format_number, load_video, img_save, make_or_load_model, save_model

# Initialize stepper

stepper = make_or_load_model(Stepper, 'stepper', Encoder(), Decoder()).to(device)

# Load video

frames = load_video(video_path, start=0, num_frames=100, num_frames_per_frame=20)

dataloader = build_stepper_dataloader(frames)

# Initialize trainer

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(stepper.parameters(), lr=3e-4)

def after_batch(trainer, firing):
    
    print(f'Epoch: {trainer.epoch_idx}, Batch: {trainer.batch_idx}, Epoch Loss: {format_number(trainer.epoch_loss)}, Batch Loss: {format_number(trainer.batch_loss)}')

    if firing:
        img_save(trainer.batch_xs[0][-1], 'input.png')
        img_save(trainer.batch_outputs[0][-1], 'output.png')

def after_epoch(trainer, firing):

    if trainer.epoch_idx == 0 or trainer.epoch_loss < min(trainer.epoch_losses):
        save_model(trainer.model, "stepper", trainer.epoch_loss)
    
    if firing:
        print(f'Epoch: {trainer.epoch_idx}, Epoch Loss: {trainer.epoch_loss}')

trainer = Trainer(stepper, dataloader, criterion, optimizer, after_batch, after_epoch)

# Train stepper

trainer.fit(20)