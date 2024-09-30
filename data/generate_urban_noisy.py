import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import torchaudio
from tqdm import tqdm
import soundfile as sf

# 設定
target_folder_train = "/Users/rockwell/Documents/python/FSSE/data/source/train/clean"
target_folder_test = "/Users/rockwell/Documents/python/FSSE/data/source/test/clean"
UrbanNoiseDir = "/Users/rockwell/Documents/python/FSSE/data/source/urban/"
noise_class_dictionary = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}


# 音声ファイルにノイズを追加する関数
def add_noise_to_audio(base_audio, noise_audio, snr):
    base_audio = base_audio.numpy()[0]  # base_audioをnumpy配列に変換
    noise_audio = noise_audio.numpy()[0]  # noise_audioをnumpy配列に変換

    # ノイズの長さが短い場合、ループ再生で長さを揃える
    if len(noise_audio) < len(base_audio):
        noise_audio = np.tile(noise_audio, int(np.ceil(len(base_audio) / len(noise_audio))))[: len(base_audio)]
    else:
        noise_audio = noise_audio[: len(base_audio)]  # 長い場合は切り捨て

    # パワー計算
    signal_power = np.mean(base_audio**2)
    noise_power = np.mean(noise_audio**2)

    # SNRに応じてノイズの振幅を調整
    target_noise_power = signal_power / (10 ** (snr / 10))
    noise_scaling_factor = np.sqrt(target_noise_power / noise_power)
    noise_audio = noise_audio * noise_scaling_factor

    # ノイズと信号を合成
    noisy_audio = base_audio + noise_audio[: len(base_audio)]
    return np.clip(noisy_audio, -1.0, 1.0).astype(np.float32)  # [-1, 1]にクリップ


# 単一のノイズタイプでノイズを付加する関数
def makeCorruptedFile_singletype(filename, base_folder, dest, noise_type, snr):
    true_path = os.path.join(base_folder, filename)
    if not os.path.exists(true_path):
        print(f"File {true_path} does not exist.")
        return
    try:
        base_audio, base_sr = torchaudio.load(true_path)  # torchaudioで音声を読み込む

        # 指定されたノイズタイプのフォルダからノイズを選択
        noise_dir = os.path.join(UrbanNoiseDir, str(noise_type) + "_" + noise_class_dictionary[noise_type])
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]

        if len(noise_files) == 0:
            print(f"No noise files found for noise_type {noise_type} in {noise_dir}")
            return

        noise_file = random.choice(noise_files)

        # ノイズをロード
        noise_path = os.path.join(noise_dir, noise_file)
        noise_audio, noise_sr = torchaudio.load(noise_path)  # torchaudioでノイズを読み込む

        # リサンプリング（必要であれば）
        if base_sr != noise_sr:
            noise_audio = torchaudio.transforms.Resample(orig_freq=noise_sr, new_freq=base_sr)(noise_audio)

        # ノイズを付加
        combined = add_noise_to_audio(base_audio, noise_audio, snr)
        target_dest = os.path.join(dest, filename)

        # 保存
        sf.write(target_dest, combined, base_sr)
        # print(f"Processed and saved: {target_dest}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")


# ディレクトリ内の全てのファイルにノイズを追加する関数
def process_directory(input_dir, output_dir, noise_type, num_workers=10):
    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    # マルチスレッドで処理を実行
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename in files:
            # print(f"Processing file: {filename}")

            snr = random.randint(0, 10)  # ランダムなSNRを指定
            futures.append(
                executor.submit(makeCorruptedFile_singletype, filename, input_dir, output_dir, noise_type, snr)
            )

        # tqdmで進行状況を同期させる
        for future in tqdm(futures):
            future.result()  # 各スレッドの結果を待機してエラーチェックを行う


if __name__ == "__main__":
    # ノイズの種類を選択
    print("ノイズの種類を選択してください:")
    for key, value in noise_class_dictionary.items():
        print(f"\t{key}: {value}")

    noise_type = int(input("ノイズタイプを選んでください (0-9): "))

    # トレーニングデータ用のディレクトリを作成
    inp_folder_train = f"/Users/rockwell/Documents/python/FSSE/data/source/train/noisy/urban-{noise_type}"
    os.makedirs(inp_folder_train, exist_ok=True)

    # テストデータ用のディレクトリを作成
    inp_folder_test = f"/Users/rockwell/Documents/python/FSSE/data/source/test/noisy/urban-{noise_type}"
    os.makedirs(inp_folder_test, exist_ok=True)

    # トレーニングデータの処理
    print("トレーニングデータを生成中...")
    process_directory(target_folder_train, inp_folder_train, noise_type)

    # テストデータの処理
    print("テストデータを生成中...")
    process_directory(target_folder_test, inp_folder_test, noise_type)
