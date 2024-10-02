import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from data.data_module import FSSEDataModule
from models.dcunet import DCUnet10
from models.model import DN
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme
from lightning.pytorch.strategies import DDPStrategy
import torch
import torch.multiprocessing as mp

# Constants
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

# Set multiprocessing and precision
mp.set_start_method("spawn", force=True)
torch.set_float32_matmul_precision("high")

def train_clean_to_clean():
    # データモジュールの作成
    datamodule = FSSEDataModule(
        SAMPLE_RATE=SAMPLE_RATE,
        N_FFT=N_FFT,
        HOP_LENGTH=HOP_LENGTH,
        data_dir="/workspace/app/FSSE/data/source",
        batch_size=32,
    )
    datamodule.setup()

    # モデルのロード
    model = DN(dim=32)

    # clean-to-clean用のデータローダーを取得
    clean_train_loader = datamodule.train_clean_dataloader()
    clean_test_loader = datamodule.val_clean_dataloader()

    # チェックポイントとログの設定
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/workspace/app/FSSE/checkpoints/clean_to_clean",
        filename="model-{epoch:02d}-{step:04d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
    )
    logger = TensorBoardLogger("/workspace/app/FSSE/tb_logs/clean_to_clean", name="my_model")
    
    # Early stopping の設定
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=20, verbose=True, mode="min")
    strategy = strategy = DDPStrategy(find_unused_parameters=True)

    # プログレスバーの設定
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

    # トレーナーの設定
    trainer = Trainer(
        accelerator="cuda",
        callbacks=[checkpoint_callback, progress_bar, early_stopping_callback],
        logger=logger,
        max_epochs=800,
        strategy=strategy,
        devices=[0,1,2,3],
    )

    # clean-to-clean モデルの学習
    trainer.fit(model, clean_train_loader, clean_test_loader)

    # モデルの重みを保存
    trainer.save_checkpoint("/workspace/app/FSSE/checkpoints/clean_to_clean_model.ckpt")


if __name__ == "__main__":
    train_clean_to_clean()

