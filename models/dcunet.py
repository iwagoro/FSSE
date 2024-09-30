import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lightning.pytorch import LightningModule
import torch
from .Encoder import Encoder
from .Decoder import Decoder
from utils.metrics import getPesqList, getSNRList, getSTOIList
import torchaudio
from pathlib import Path
from utils.stft import istft, tensor_stft, tensor_istft
from utils.loss import basic_loss, reg_loss, wsdr_loss
from utils.subsample import subsample2


class DCUnet10(LightningModule):
    def __init__(self, dataset="", loss_type=""):
        super().__init__()

        self.pesqNb_scores = []
        self.pesqWb_scores = []
        self.snr_scores = []
        self.stoi_scores = []
        self.total_samples = 0
        self.saved = False
        self.loss_type = loss_type

        self.model = "dcunet"
        self.dataset = dataset
        self.gamma = self.gamma = 1

        self.save_hyperparameters()
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(3, 3), stride_size=(2, 1), in_channels=90, out_channels=90)

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3, 3), stride_size=(2, 1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(3, 3), stride_size=(2, 2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(
            filter_size=(3, 3),
            stride_size=(2, 2),
            in_channels=90,
            output_padding=(1, 1),
            out_channels=1,
            last_layer=True,
        )

    def forward(self, x, is_istft=True):
        if isinstance(x, list):
            x = torch.stack(x)
        # downsampling/encoding
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        # upsampling/decoding
        u0 = self.upsample0(d4)
        # skip-connection
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)

        # u4 - the mask
        output = u4 * x

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        pred = tensor_istft(pred)
        loss = wsdr_loss(x, pred, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        pred = tensor_istft(pred)
        loss = wsdr_loss(x, pred, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
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
        Path("pred/" + self.model + "-" + self.dataset).mkdir(parents=True, exist_ok=True)
        for i in range(len(x)):
            x_audio = istft(x[i])
            y_audio = istft(y[i])
            pred_audio = istft(pred[i])

            # Save the audio files
            torchaudio.save(
                "./pred/" + self.model + "-" + self.dataset + "/noisy" + str(i) + ".wav", x_audio.cpu(), 48000
            )
            torchaudio.save(
                "./pred/" + self.model + "-" + self.dataset + "/clean" + str(i) + ".wav", y_audio.cpu(), 48000
            )
            torchaudio.save(
                "./pred/" + self.model + "-" + self.dataset + "/pred" + str(i) + ".wav", pred_audio.cpu(), 48000
            )

    def on_predict_end(self):
        average_pesqNb = sum(self.pesqNb_scores) / self.total_samples
        average_pesqWb = sum(self.pesqWb_scores) / self.total_samples
        average_snr = sum(self.snr_scores) / self.total_samples
        average_stoi = sum(self.stoi_scores) / self.total_samples

        print("-----------------------------------")
        print("model : " + self.model)
        print("dataset : " + self.dataset)
        print("-----------------------------------")
        print(f"pesq-nb :{average_pesqNb}")
        print(f"pesq-wb :{average_pesqWb}")
        print(f"snr : {average_snr}")
        print(f"stoi : {average_stoi}")

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=8e-4)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=8e-4)
