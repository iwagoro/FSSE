import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from lightning.pytorch import Trainer
from models.maml_module import MAMLModule
from models.dcunet import DCUnet10
from data.data_module_meta import FSSEDataModule

torch.set_float32_matmul_precision('medium')

def main():
    # データモジュールの初期化
    noise_types = ['white', 'urban-0', 'urban-1','urban-2','urban-3','urban-4','urban-5','urban-6','urban-7']  # 使用するノイズタイプ
    data_module = FSSEDataModule(
        SAMPLE_RATE=48000,
        N_FFT=1022,
        HOP_LENGTH=256,
        noise_types=noise_types,
        data_dir="/workspace/app/FSSE/data/source",
        batch_size=4,  # MAMLでは小さなバッチサイズを使用
        few_shot_k=60
    )
    data_module.setup()

    # モデルの初期化
    model = DCUnet10()
    maml_model = MAMLModule(model=model)

    # トレーナーの初期化
    trainer = Trainer(
        max_epochs=10,
        devices=1  # 使用するGPU数
    )

    # トレーニングの実行
    trainer.fit(maml_model, datamodule=data_module)

if __name__ == '__main__':
    main()
