import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lightning.pytorch import LightningModule
import torch.nn as nn
import torch
from .CConv import CConv2d, CConvTranspose2d
import torchaudio
import torch.optim as optim
from pathlib import Path
from utils.stft import istft
from utils.loss import wsdr_loss

from utils.metrics import getPesqList, getSNRList, getSTOIList


class Encoder(nn.Module):
    def __init__(self, in_channels, n_feat):
        super(Encoder, self).__init__()

        self.f0 = nn.Sequential(
            CConv2d(in_channels, n_feat, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
            CConv2d(n_feat, n_feat, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )

        self.f1 = nn.Sequential(
            CConv2d(n_feat, n_feat, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
            CConv2d(n_feat, n_feat, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )

        self.f2 = nn.Sequential(
            CConv2d(n_feat, n_feat, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
            CConv2d(n_feat, n_feat, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )

        self.f3 = nn.Sequential(
            CConv2d(n_feat, n_feat, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
            CConv2d(n_feat, n_feat, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )

        self.f4 = nn.Sequential(
            CConv2d(n_feat, n_feat, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
            CConv2d(n_feat, n_feat, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        f0 = self.f0(x)
        f1 = self.f1(f0)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = self.f4(f3)

        return [x, f0, f1, f2, f3, f4]


class Decoder(nn.Module):
    def __init__(self, dim, out_channels):
        super(Decoder, self).__init__()

        self.d0 = nn.Sequential(
            CConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            CConv2d(dim, dim, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
        )

        self.d1 = nn.Sequential(
            CConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            CConv2d(dim, dim, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
        )

        self.d2 = nn.Sequential(
            CConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            CConv2d(dim, dim, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
        )

        self.d3 = nn.Sequential(
            CConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            CConv2d(dim, dim, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
        )

        self.d4 = nn.Sequential(
            CConvTranspose2d(dim, dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(inplace=True),
            CConv2d(dim, dim, kernel_size=(7, 5), stride=1, padding=(3, 2)),
            nn.LeakyReLU(inplace=True),
        )

        self.out = CConv2d(dim, out_channels, kernel_size=(7, 5), stride=1, padding=(3, 2))

    def forward(self, x):
        d0 = self.d0(x[5])
        d1 = self.d1(d0 + x[4])
        d2 = self.d2(d1 + x[3])
        d3 = self.d3(d2 + x[2])
        d4 = self.d4(d3 + x[1])

        out = self.out(d4) + x[0]

        return out


class MSFR(nn.Module):
    """Multi-Scale Feature Recursive module"""

    def __init__(self, in_c, out_c, stride=1):
        super(MSFR, self).__init__()
        in_channels = out_c
        s = 0.5
        self.num_steps = 2
        self.two = int(in_channels * s)
        self.four = int(in_channels * (s**2))
        self.eight = int(in_channels * (s**3))
        self.sixteen = int(in_channels * (s**4))

        self.inputs_c0 = nn.Sequential(
            CConv2d(in_c, in_channels, kernel_size=(7, 5), stride=1, padding=(3, 2)), nn.LeakyReLU(inplace=True)
        )

        self.c1 = nn.Sequential(
            CConv2d(self.two, self.two, kernel_size=(9, 7), stride=stride, padding=(4, 3)), nn.LeakyReLU(inplace=True)
        )

        self.c2 = nn.Sequential(
            CConv2d(self.four, self.four, kernel_size=(7, 5), stride=stride, padding=(3, 2)), nn.LeakyReLU(inplace=True)
        )

        self.c3 = nn.Sequential(
            CConv2d(self.eight, self.eight, kernel_size=(5, 3), stride=stride, padding=(2, 1)), nn.LeakyReLU(inplace=True)
        )

        self.c4 = nn.Sequential(
            CConv2d(self.sixteen, self.sixteen, kernel_size=(3, 3), stride=stride, padding=(1, 1)), nn.LeakyReLU(inplace=True)
        )

        self.out = CConv2d(self.num_steps * in_channels, in_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        feature = []
        for _ in range(self.num_steps):
            out_c0 = self.inputs_c0(inputs)
            distilled_c0, remaining_c0 = torch.split(out_c0, (self.two, self.two), dim=1)

            out_c1 = self.c1(remaining_c0)
            distilled_c1, remaining_c1 = torch.split(out_c1, (self.four, self.four), dim=1)

            out_c2 = self.c2(remaining_c1)
            distilled_c2, remaining_c2 = torch.split(out_c2, (self.eight, self.eight), dim=1)

            out_c3 = self.c3(remaining_c2)
            distilled_c3, remaining_c3 = torch.split(out_c3, (self.sixteen, self.sixteen), dim=1)

            out_c4 = self.c4(remaining_c3)

            out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4, distilled_c0], dim=1)

            inputs = out
            feature.append(out)

        out_fused = torch.cat(feature, dim=1)

        return self.out(out_fused)


class Noiser(nn.Module):
    def __init__(self, dim, in_channels=1, out_channels=1):
        super(Noiser, self).__init__()

        self.f0 = nn.Sequential(
            CConv2d(in_channels, dim, kernel_size=(7, 5), stride=1, padding=(3, 2)), nn.LeakyReLU(inplace=True)
        )
        self.F = MSFR(dim, dim, 1)

        self.f1 = MSFR(dim, dim, 1)
        self.f2 = MSFR(dim, dim, 1)
        self.f3 = MSFR(dim, dim, 1)
        self.f4 = MSFR(dim, dim, 1)

    def forward(self, x, F):
        f0 = self.F(self.f0(x))
        f1 = self.f1(f0 + F[1])
        f2 = self.f2(f1 + F[2])
        f3 = self.f3(f2 + F[3])
        f4 = self.f4(f3 + F[4])

        return [x, f0, f1, f2, f3, f4]


class DN(LightningModule):
    def __init__(self, dim, in_channels=1, out_channels=1, few_shot=False):
        super(DN, self).__init__()

        self.pesqNb_scores = []
        self.pesqWb_scores = []
        self.snr_scores = []
        self.stoi_scores = []
        self.total_samples = 0

        self.few_shot = few_shot
        self.Encoder = Encoder(in_channels, dim)

        if self.few_shot:
            for p in self.parameters():
                p.requires_grad = False

        self.Decoder = Decoder(dim, out_channels)
        self.Noiser = Noiser(dim, in_channels, out_channels)

    def forward(self, x):

        few_shot = self.few_shot

        fea = self.Encoder(x)

        if few_shot:
            fea_N = self.Noiser(x, fea)
            out = self.Decoder(fea_N)
        else:
            out = self.Decoder(fea)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = nn.MSELoss()(y_hat, y)
        loss = wsdr_loss(x,y_hat,y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = nn.MSELoss()(y_hat, y)
        loss = wsdr_loss(x,y_hat,y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        pred = self(x)

        pesqNb = getPesqList(pred, y, "nb")
        pesqWb = getPesqList(pred, y, "wb")
        snr = getSNRList(pred, y)
        stoi = getSTOIList(pred, y)

        self.pesqNb_scores.append(pesqNb)
        self.pesqWb_scores.append(pesqWb)
        self.snr_scores.append(snr)
        self.stoi_scores.append(stoi)
        self.total_samples += batch[0].size(0)
        
        # Ensure the 'pred' directory exists
        Path("/workspace/app/FSSE/predictions").mkdir(parents=True, exist_ok=True)
        for i in range(len(x)):
            x_audio = istft(x[i])
            y_audio = istft(y[i])
            pred_audio = istft(pred[i])
            
            # Save the audio files
            torchaudio.save("/workspace/app/FSSE/predictions/" +  "noisy"+str(i)+".wav", x_audio.cpu(), 48000)
            torchaudio.save("/workspace/app/FSSE/predictions/" +  "clean"+str(i)+".wav", y_audio.cpu(), 48000)
            torchaudio.save("/workspace/app/FSSE/predictions/" +  "pred"+str(i)+".wav", pred_audio.cpu(), 48000)

    def on_predict_end(self):
        average_pesqNb = sum(self.pesqNb_scores) / self.total_samples
        average_pesqWb = sum(self.pesqWb_scores) / self.total_samples
        average_snr = sum(self.snr_scores) / self.total_samples
        average_stoi = sum(self.stoi_scores) / self.total_samples

        print("-----------------------------------")
        print(f"pesq-nb :{average_pesqNb}")
        print(f"pesq-wb :{average_pesqWb}")
        print(f"snr : {average_snr}")
        print(f"stoi : {average_stoi}")

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
        scheduler = {
            # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            # 'scheduler' : optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5),
            'scheduler' : optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=8e-3,        # 最大学習率
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
