import torch
import torch.nn as nn

from settings import *

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(frame_channels, frame_channels, (7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(frame_channels, chunk_channels, (chunk_height, chunk_width), stride=(chunk_height // 2, chunk_width // 2), padding=0),
            nn.ReLU(),
        )

    def forward(self, frames):

        original_shape = frames.shape

        frames = frames.reshape(-1, frame_channels, frame_height, frame_width)
        
        chunks = self.encoder(frames)

        return chunks.reshape(*original_shape[:-3], chunk_channels, chunkified_height, chunkified_width)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(chunk_channels, frame_channels, (chunk_height, chunk_width), stride=(chunk_height // 2, chunk_width // 2), padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(frame_channels, frame_channels, (7, 7), stride=1, padding=3),
            nn.ReLU(),
        )

    def forward(self, chunks):

        original_shape = chunks.shape
        
        chunks = chunks.reshape(-1, chunk_channels, chunkified_height, chunkified_width)
        
        frames = self.decoder(chunks)
        
        return frames.reshape(*original_shape[:-3], frame_channels, frame_height, frame_width)
    
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, frames):

        chunks = self.encoder(frames)

        results = self.decoder(chunks)

        return results