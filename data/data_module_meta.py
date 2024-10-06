
import torchaudio
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from glob import glob
import os
import torch
import random
import warnings
warnings.filterwarnings("ignore")

N_FFT = 1022
SAMPLE_RATE = 48000
HOP_LENGTH = 256
MAX_LEN = 65280

# 自前のstft関数
def stft(wav):
    window = torch.hann_window(N_FFT)
    stft = torch.stft(
        input=wav, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, return_complex=True, window=window
    )
    stft = torch.view_as_real(stft)
    return stft

def istft(stft):
    window = torch.hann_window(N_FFT)
    stft = torch.view_as_complex(stft)
    wav = torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=window)
    return wav

# Few-shot用のNoisy-to-Cleanデータセット
class FewShotNoisyAudioDataset(Dataset):
    def __init__(self, noisy_files, clean_files, few_shot_k=60):
        self.noisy_files = noisy_files[:few_shot_k]
        self.clean_files = clean_files[:few_shot_k]

    def _load_sample(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        return waveform

    def _prepare_sample(self, waveform):
        channels, current_len = waveform.shape
        output = torch.zeros((channels, MAX_LEN), dtype=torch.float32)
        output[:, -min(current_len, MAX_LEN):] = waveform[:, :min(current_len, MAX_LEN)]
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

class MultiNoiseAudioDataset(FewShotNoisyAudioDataset):
    def __init__(self, noisy_files_dict, clean_files, few_shot_k=60, support_size=5, query_size=15):
        super().__init__(noisy_files=[], clean_files=[], few_shot_k=few_shot_k)
        self.noisy_files_dict = noisy_files_dict
        self.clean_files = clean_files
        self.support_size = support_size
        self.query_size = query_size

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # ノイズタイプをランダムに選択
        noise_type = random.choice(list(self.noisy_files_dict.keys()))
        noisy_files = self.noisy_files_dict[noise_type]

        # サポートセットとクエリセットを作成
        indices = random.sample(range(len(noisy_files)), self.support_size + self.query_size)
        support_indices = indices[:self.support_size]
        query_indices = indices[self.support_size:]

        x_support_noisy, y_support = self._get_samples(support_indices, noisy_files)
        x_query_noisy, y_query = self._get_samples(query_indices, noisy_files)

        return x_support_noisy, y_support, x_query_noisy, y_query
    
    def _get_samples(self, indices, noisy_files):
        x_noisy_list = []
        y_clean_list = []
        for idx in indices:
            x_clean = self._load_sample(self.clean_files[idx % len(self.clean_files)])
            x_noisy = self._load_sample(noisy_files[idx % len(noisy_files)])
            x_clean = self._prepare_sample(x_clean)
            x_noisy = self._prepare_sample(x_noisy)
            x_clean_stft = self._stft(x_clean)
            x_noisy_stft = self._stft(x_noisy)
            x_noisy_list.append(x_noisy_stft)
            y_clean_list.append(x_clean_stft)
        x_noisy_tensor = torch.stack(x_noisy_list)
        y_clean_tensor = torch.stack(y_clean_list)
        return x_noisy_tensor, y_clean_tensor


class FSSEDataModule(pl.LightningDataModule):
    def __init__(self, SAMPLE_RATE, N_FFT, HOP_LENGTH, noise_types=["white"], data_dir="/workspace/app/FSSE/data/source", batch_size=32, few_shot_k=60):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.SAMPLE_RATE = SAMPLE_RATE
        self.N_FFT = N_FFT
        self.HOP_LENGTH = HOP_LENGTH
        self.few_shot_k = few_shot_k
        self.noise_types = noise_types
    
    # def prepare_meta_batches(self):
    #     meta_batches = []
    #     for noise_type in self.noise_types:
    #         dataloader = self.get_task_dataloader(noise_type)
    #         meta_batches.append(dataloader)
    #     return meta_batches
    
    def prepare_data(self):
        # ここでデータのダウンロードや前処理を行う
        pass

    def setup(self, stage=None):
        self.train_clean = sorted(glob(os.path.join(self.data_dir, "train", "clean", "*.wav")))
        self.test_clean = sorted(glob(os.path.join(self.data_dir, "test", "clean", "*.wav")))
        
        print(len(self.train_clean))

        # ノイズタイプごとにファイルを取得
        self.train_noisy_dict = {}
        self.test_noisy_dict = {}
        for noise_type in self.noise_types:
            self.train_noisy_dict[noise_type] = sorted(glob(os.path.join(self.data_dir, "train", "noisy", noise_type, "*.wav")))
            self.test_noisy_dict[noise_type] = sorted(glob(os.path.join(self.data_dir, "test", "noisy", noise_type, "*.wav")))

        # データセットの作成
        self.train_dataset = MultiNoiseAudioDataset(self.train_noisy_dict, self.train_clean, self.few_shot_k)
        self.test_dataset = MultiNoiseAudioDataset(self.test_noisy_dict, self.test_clean, self.few_shot_k)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, persistent_workers=True)

    def get_task_dataloader(self, noise_type):
        noisy_files = self.train_noisy_dict[noise_type]
        dataset = FewShotNoisyAudioDataset(noisy_files, self.train_clean, self.few_shot_k)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    

