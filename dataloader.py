from torch.utils.data import Dataset, DataLoader
from settings import autoencoder_batch_size, stepper_batch_size, stepper_window_size

class AutoencoderDataset(Dataset):

    def __init__(self, frames, batch_size):
        self.batch_size = batch_size
        self.frames = frames[:(len(frames) // batch_size) * batch_size]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        selected_frame = self.frames[idx]
        return selected_frame, selected_frame

def build_autoencoder_dataloader(frames, batch_size=autoencoder_batch_size):
    dataset = AutoencoderDataset(frames, batch_size)
    return DataLoader(dataset, batch_size, shuffle=True)

class StepperDataset(Dataset):

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames) - stepper_window_size

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + stepper_window_size

        return self.frames[start_idx:end_idx], self.frames[start_idx+1:end_idx+1]

def build_stepper_dataloader(frames, batch_size=stepper_batch_size):
    dataset = StepperDataset(frames)
    return DataLoader(dataset, batch_size, shuffle=False)
