import torchaudio
import torch
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def add_white_noise(waveform, noise_level=0.005):
    noise = torch.randn(waveform.size()) * noise_level
    noisy_waveform = waveform + noise
    return noisy_waveform


def load_and_add_noise(file_path, noise_level=0.005):
    """
    音声ファイルを読み込み、ホワイトノイズを加える関数
    :param file_path: 音声ファイルのパス
    :param noise_level: ノイズの強さ
    :return: ホワイトノイズが加えられた音声の波形とサンプルレート
    """
    waveform, sample_rate = torchaudio.load(file_path, backend="soundfile")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    noisy_waveform = add_white_noise(waveform, noise_level)

    return noisy_waveform, sample_rate


def save_waveform(waveform, sample_rate, output_file):
    """
    音声波形をファイルに保存する関数
    :param waveform: 音声の波形
    :param sample_rate: サンプルレート
    :param output_file: 出力ファイルのパス
    """
    torchaudio.save(output_file, waveform, sample_rate)


def process_file(file_path, output_dir, noise_level):
    """
    1つの音声ファイルを処理し、ホワイトノイズを加えて保存する関数
    :param file_path: 音声ファイルのパス
    :param output_dir: 出力ディレクトリ
    :param noise_level: ノイズの強さ
    """
    noisy_waveform, sample_rate = load_and_add_noise(file_path, noise_level)
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    save_waveform(noisy_waveform, sample_rate, output_file)


def process_directory_multithread(input_dir, output_dir, noise_level=0.005, max_workers=4):
    """
    マルチスレッドで指定されたディレクトリ内のすべてのwavファイルにホワイトノイズを加えて保存する関数
    :param input_dir: 入力ディレクトリのパス
    :param output_dir: 出力ディレクトリのパス
    :param noise_level: ノイズの強さ
    :param max_workers: 最大スレッド数
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ディレクトリ内のすべてのwavファイルを取得
    wav_files = glob(os.path.join(input_dir, "*.wav"))

    # tqdmを使用して進捗バーを表示
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, wav_file, output_dir, noise_level): wav_file for wav_file in wav_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                future.result()
            except Exception as exc:
                print(f"File {futures[future]} generated an exception: {exc}")


# 使用例
input_directory = "/Users/rockwell/Documents/python/FSSE/data/source/train/clean"  # 入力wavファイルがあるディレクトリ
output_directory = "/Users/rockwell/Documents/python/FSSE/data/source/train/noisy/white"  # ノイズを加えたwavファイルを保存するディレクトリ
noise_level = 0.01  # ノイズの強さ
max_workers = 4  # 使用するスレッド数

# ディレクトリ内のすべてのwavファイルにホワイトノイズをマルチスレッドで加える
process_directory_multithread(input_directory, output_directory, noise_level, max_workers)

print(f"すべてのファイルにホワイトノイズをマルチスレッドで加えて保存しました。")
