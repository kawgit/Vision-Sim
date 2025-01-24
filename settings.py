video_path = 'videos/screensaver.mp4'

frame_width = 640
frame_height = 360
frame_channels = 3

chunk_width = 40
chunk_height = 40
chunk_channels = 8

chunkified_width = frame_width // chunk_width * 2 - 1
chunkified_height = frame_height // chunk_height * 2 - 1
compression_rate = (frame_width * frame_height * frame_channels) // (chunkified_width * chunkified_height * chunk_channels)

autoencoder_batch_size = 16

stepper_hidden_channels = 32

stepper_window_size = 2

stepper_batch_size = 8
