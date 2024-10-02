import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_module import FSSEDataModule
from models.model import DN
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme
import torch
import torch.multiprocessing as mp

# Constants
SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

# Set multiprocessing and precision
mp.set_start_method("spawn", force=True)
torch.set_float32_matmul_precision("high")

def train_few_shot_noisy_to_clean():
    # データモジュールの作成
    datamodule = FSSEDataModule(
        SAMPLE_RATE=SAMPLE_RATE,
        N_FFT=N_FFT,
        HOP_LENGTH=HOP_LENGTH,
        data_dir="/workspace/app/FSSE/data/source",
        batch_size=1,
    )
    datamodule.setup()

    # モデルのロード
    model = DN(dim=32,few_shot=True)
    checkpoint = torch.load("/workspace/app/FSSE/checkpoints/StepLR.ckpt")
    model.load_state_dict(checkpoint["state_dict"])

    # few-shot Noisy-to-Clean用のデータローダーを取得
    few_shot_train_loader = datamodule.train_few_shot_dataloader()
    few_shot_val_loader = datamodule.val_few_shot_dataloader()

    # チェックポイントとログの設定
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/workspace/app/FSSE/checkpoints/few_shot_noisy_to_clean",
        filename="model-{epoch:02d}-{step:04d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
    )
    logger = TensorBoardLogger("/workspace/app/FSSE/tb_logs/few_shot_noisy_to_clean", name="my_model")

    # Early stopping の設定
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")

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
    few_shot_trainer = Trainer(
        accelerator="cuda",
        max_epochs=800,
        callbacks=[checkpoint_callback, progress_bar, early_stopping_callback],
        logger=logger,
        devices=1,
    )

    # few-shot Noisy-to-Clean の学習
    few_shot_trainer.fit(model, few_shot_train_loader, few_shot_val_loader)

    # モデルの重みを保存
    torch.save(model.state_dict(), "/workspace/app/FSSE/checkpoints/few_shot_noisy_to_clean_model.ckpt")


if __name__ == "__main__":
    train_few_shot_noisy_to_clean()
