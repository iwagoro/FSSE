import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from data.data_module import FSSEDataModule
from models.dcunet import DCUnet10
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Constants
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

# Set multiprocessing and precision
mp.set_start_method("spawn", force=True)
torch.set_float32_matmul_precision("high")


def main():
    # Fixed parameters
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 8
    loss_type = "nct"  # Only 'nct' loss type is used
    model_type = "dcunet"  # Model fixed to 'dcunet'
    max_epochs = 150
    devices = [0]  # GPU devices to use

    # Create the data module
    datamodule = FSSEDataModule(
        SAMPLE_RATE=SAMPLE_RATE,
        N_FFT=N_FFT,
        HOP_LENGTH=HOP_LENGTH,
        data_dir="/Users/rockwell/Documents/python/FSSE/data/source",
        batch_size=TRAIN_BATCH_SIZE,
    )

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    # Update checkpoint and logger paths
    checkpoint_dir = f"./checkpoints/{model_type}-white"
    tb_log_dir = f"tb_logs/{model_type}-white"

    # Checkpoint and logging setup
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="model-{epoch:02d}-{step:04d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
    )

    logger = TensorBoardLogger(tb_log_dir, name="my_model")
    strategy = DDPStrategy(find_unused_parameters=True)

    # Progress bar setup
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green_yellow",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="cyan",
            processing_speed="#ff1493",
            metrics="#ff1493",
            metrics_text_delimiter="\n",
        )
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")

    # Initialize the model
    model = DCUnet10(loss_type=loss_type)

    # Create the trainer and start training
    # trainer = Trainer(
    #     accelerator="mps",
    #     callbacks=[checkpoint_callback, progress_bar],
    #     logger=logger,
    #     max_epochs=max_epochs,
    #     strategy=strategy,
    #     devices=devices,
    # )

    trainer = Trainer(
        accelerator="mps",  # MPS（Apple SiliconのGPU）で実行
        callbacks=[checkpoint_callback, progress_bar],
        logger=logger,
        max_epochs=max_epochs,
        devices=1,  # MPSの場合、デバイス数は1
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
