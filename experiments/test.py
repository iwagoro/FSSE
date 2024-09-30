import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.data_module import FSSEDataModule

SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

datamodule = FSSEDataModule(
    SAMPLE_RATE=SAMPLE_RATE,
    N_FFT=N_FFT,
    HOP_LENGTH=HOP_LENGTH,
    data_dir="/Users/rockwell/Documents/python/FSSE/data/source",
    batch_size=32,
)

datamodule.setup()
train_dataloader = datamodule.train_dataloader()

print(len(train_dataloader.dataset))
