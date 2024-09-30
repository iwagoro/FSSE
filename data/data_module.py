import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from glob import glob
import os
import torch

N_FFT = 1022
HOP_LENGTH = 256
MAX_LEN = 65280


# 自前のstft関数
def stft(wav):
    window = torch.hann_window(N_FFT, device=wav.device)
    stft = torch.stft(
        input=wav, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, return_complex=True, window=window
    )
    stft = torch.view_as_real(stft)
    return stft


def istft(stft):
    window = torch.hann_window(N_FFT, device=stft.device)
    stft = torch.view_as_complex(stft)
    wav = torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=window)
    return wav


def tensor_stft(wav):
    result = []
    for i in range(len(wav)):
        wav_stft = stft(wav[i])
        result.append(wav_stft)
    result = torch.cat(result, dim=0).unsqueeze(1)
    return result


def tensor_istft(stft):
    result = []
    for i in range(len(stft)):
        wav = istft(stft[i])
        result.append(wav)
    result = torch.cat(result, dim=0).unsqueeze(1)
    return result


# Datasetクラス
class AudioDataset(Dataset):
    def __init__(self, noisy_files, clean_files):
        self.noisy_files = noisy_files
        self.clean_files = clean_files

    def _load_sample(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        return waveform

    def _prepare_sample(self, waveform):
        channels, current_len = waveform.shape
        output = torch.zeros((channels, MAX_LEN), dtype=torch.float32, device=waveform.device)
        output[:, -min(current_len, MAX_LEN) :] = waveform[:, : min(current_len, MAX_LEN)]
        return output

    def _stft(self, waveform):
        return stft(waveform)

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        # 音声データを読み込み
        x_clean = self._load_sample(self.clean_files[idx])
        x_noisy = self._load_sample(self.noisy_files[idx])

        # 音声データを準備（パディング）
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)

        # STFTを適用
        x_noisy_stft = self._stft(x_noisy)
        x_clean_stft = self._stft(x_clean)

        return x_noisy_stft, x_clean_stft


class FSSEDataModule(pl.LightningDataModule):
    def __init__(self, SAMPLE_RATE, N_FFT, HOP_LENGTH, data_dir="./source", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.SAMPLE_RATE = SAMPLE_RATE
        self.N_FFT = N_FFT
        self.HOP_LENGTH = HOP_LENGTH

    def setup(self, stage=None):
        # 訓練データセット
        self.train_noisy = sorted(glob(os.path.join(self.data_dir, "train", "noisy", "white", "*.wav")))
        self.train_clean = sorted(glob(os.path.join(self.data_dir, "train", "clean", "*.wav")))

        # テストデータセット
        self.test_noisy = sorted(glob(os.path.join(self.data_dir, "test", "noisy", "white", "*.wav")))
        self.test_clean = sorted(glob(os.path.join(self.data_dir, "test", "clean", "*.wav")))

        # 訓練・テストデータセットのインスタンス
        self.train_dataset = AudioDataset(self.train_noisy, self.train_clean)
        self.test_dataset = AudioDataset(self.test_noisy, self.test_clean)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=7)
