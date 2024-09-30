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

    # データモジュールの作成
    datamodule = FSSEDataModule(
        SAMPLE_RATE=SAMPLE_RATE,
        N_FFT=N_FFT,
        HOP_LENGTH=HOP_LENGTH,
        data_dir="/workspace/app/FSSE/data/source",
        batch_size=32,
        few_shot_k=256
    )
    datamodule.setup()


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

    # モデルの初期化
    model = DN(dim=32)
    checkpoint = torch.load("")  # 予測に使用するモデルのチェックポイント
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # モデルを評価モードに

    # トレーナーの設定
    trainer = Trainer(
        accelerator="cuda",  # MPS（Apple SiliconのGPU）で実行
        callbacks=[progress_bar],
        devices=[0]  # MPSの場合、デバイス数は1
    )

    
    trainer.predict(model,dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    main()
