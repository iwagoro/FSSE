# maml_module.py

import torch
from lightning.pytorch import LightningModule
from .dcunet import DCUnet10
from utils.loss import wsdr_loss
from torch.optim import Adam

import torch.optim as optim


class MAMLModule(LightningModule):
    def __init__(self, model, inner_lr=1e-3, outer_lr=1e-4, inner_steps=5):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_support, y_support, x_query, y_query = batch
    
        # デバイスの取得
        device = self.device
    
        # テンソルの形状を取得
        batch_size, support_size, channels, height, width, complex_dim = x_support.shape
    
        # チャンネル次元と複素数次元を統合
        channels = channels * complex_dim  # 1 * 2 = 2
    
        # テンソルの形状を変換
        x_support = x_support.view(-1, channels, height, width).to(device)
        y_support = y_support.view(-1, channels, height, width).to(device)
        x_query = x_query.view(-1, channels, height, width).to(device)
        y_query = y_query.view(-1, channels, height, width).to(device)
    
        # モデルのコピーを作成
        temp_model = DCUnet10()
        temp_model.load_state_dict(self.model.state_dict())
        temp_model.to(device)
        temp_model.train()
    
        # 内側のループ（タスクごとの学習）
        inner_optimizer = Adam(temp_model.parameters(), lr=self.inner_lr)
        for _ in range(self.inner_steps):
            preds = temp_model(x_support)
            loss = wsdr_loss(x_support, preds, y_support)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
    
        # 外側のループ（メタ更新）
        preds_query = temp_model(x_query)
        loss_query = wsdr_loss(x_query, preds_query, y_query)
        self.log('meta_train_loss', loss_query)
        return loss_query




    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = {
            # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            # 'scheduler' : optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5),
            'scheduler' : optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=1e-3,        # 最大学習率
                steps_per_epoch=60,  # エポックごとのステップ数
                epochs=10,          # 学習の総エポック数
                pct_start=0.3,      # どこで最大学習率に到達するか
                anneal_strategy='cos',  # コサイン減衰
                div_factor=25,      # 最小学習率の設定
                final_div_factor=1e4  # 最終学習率の設定
            ),    
            'monitor': 'val_loss',  # 監視する指標
            'interval': 'epoch',    # スケジューラが適用されるタイミング
            'frequency': 1          # スケジューラの頻度
        }
        return [optimizer], [scheduler]
        # return optimizer
