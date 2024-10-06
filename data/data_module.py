from pytorch_lightning.utilities.types import EVAL_DATALOADERS
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


# Clean-to-Clean用のデータセット
class CleanAudioDataset(Dataset):
    def __init__(self, clean_files):
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
        return len(self.clean_files)

    def __getitem__(self, idx):
        x_clean = self._load_sample(self.clean_files[idx])
        x_clean = self._prepare_sample(x_clean)
        x_clean_stft = self._stft(x_clean)
        return x_clean_stft, x_clean_stft  # Clean-to-Cleanのペア


# Few-shot用のNoisy-to-Cleanデータセット
class FewShotNoisyAudioDataset(Dataset):
    def __init__(self, noisy_files, clean_files, few_shot_k=60):
        self.noisy_files = noisy_files[:few_shot_k]  # few-shotのため、データ量を制限
        self.clean_files = clean_files[:few_shot_k]

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
        x_clean = self._load_sample(self.clean_files[idx])
        x_noisy = self._load_sample(self.noisy_files[idx])
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        x_clean_stft = self._stft(x_clean)
        x_noisy_stft = self._stft(x_noisy)
        return x_noisy_stft, x_clean_stft  # Noisy-to-Cleanのペア


class FSSEDataModule(pl.LightningDataModule):
    def __init__(self, SAMPLE_RATE, N_FFT, HOP_LENGTH, data_dir="./source", batch_size=32, few_shot_k=60):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.SAMPLE_RATE = SAMPLE_RATE
        self.N_FFT = N_FFT
        self.HOP_LENGTH = HOP_LENGTH
        self.few_shot_k = few_shot_k  # few-shotのためのK値

    def setup(self, stage=None):
        self.train_clean = sorted(glob(os.path.join(self.data_dir, "train", "clean", "*.wav")))
        self.test_clean = sorted(glob(os.path.join(self.data_dir, "test", "clean", "*.wav")))
        self.train_noisy = sorted(glob(os.path.join(self.data_dir, "train", "noisy", "white", "*.wav")))
        self.test_noisy = sorted(glob(os.path.join(self.data_dir, "test", "noisy", "white", "*.wav")))

        # clean-to-clean
        self.clean_train_dataset = CleanAudioDataset(self.train_clean)
        self.clean_test_dataset = CleanAudioDataset(self.test_clean)

        # noisy-to-clean
        self.few_shot_train_dataset = FewShotNoisyAudioDataset(self.train_noisy, self.train_clean, self.few_shot_k)
        self.few_shot_test_dataset = FewShotNoisyAudioDataset(self.test_noisy, self.test_clean, self.few_shot_k)
        
        # prediction
        self.full_noisy_clean_test_dataset = FewShotNoisyAudioDataset(self.test_noisy, self.test_clean, self.few_shot_k)

    # clean-to-clean for training
    def train_clean_dataloader(self):
        return DataLoader(self.clean_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=27,persistent_workers=True)

    # clean-to-clean for validation
    def val_clean_dataloader(self):
        return DataLoader(self.clean_test_dataset, batch_size=self.batch_size, num_workers=27,persistent_workers=True)

    # noisy-to-clean for training
    def train_few_shot_dataloader(self):
        return DataLoader(self.few_shot_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=27,persistent_workers=True)

    # noisy-to-clean for validation
    def val_few_shot_dataloader(self):
        return DataLoader(self.full_noisy_clean_test_dataset, batch_size=self.batch_size, num_workers=27,persistent_workers=True)

    # noisy-to-clean for prediction
    def test_dataloader(self):
        return DataLoader(self.full_noisy_clean_test_dataset, batch_size=self.batch_size, num_workers=7,persistent_workers=True)

