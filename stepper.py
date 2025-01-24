import time
import torch
import torch.nn as nn

from settings import *
from utils import img_save

class Stepper(nn.Module):

    def __init__(self, encoder, decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
        self.stepper = nn.Sequential(
            nn.Conv3d(chunk_channels, stepper_hidden_channels, (2, 11, 11), stride=1, padding=(1, 5, 5)),
            nn.ReLU(),
            nn.Conv3d(stepper_hidden_channels, chunk_channels, (1, 11, 11), stride=1, padding=(0, 5, 5)),
            nn.ReLU(),
        )

    def forward(self, frames):
        
        chunks = self.encoder(frames)

        chunks = chunks.permute(0, 2, 1, 3, 4)

        chunks = self.stepper(chunks)[:, :, -stepper_window_size:, :, :]

        chunks = chunks.permute(0, 2, 1, 3, 4)

        frames = self.decoder(chunks)

        return frames
    
    def forward_single(self, frames):

        chunks = self.encoder(frames[:, -stepper_window_size:, :, :, :])

        chunks = chunks.permute(0, 2, 1, 3, 4)

        # chunks = self.stepper(chunks)

        chunks = chunks.permute(0, 2, 1, 3, 4)

        frame = self.decoder(chunks[0][-2:])

        return frame
    
    def generate(self, frames, num_frames):

        self.eval()

        for i in range(num_frames):

            new_frame = self.forward_single(frames.reshape(1, *frames.shape))
            
            frames = torch.cat((frames, new_frame))

            torch.mps.empty_cache()

            print(len(frames))
            img_save(frames[-1], 'output.png')

            time.sleep(2)

        return frames

