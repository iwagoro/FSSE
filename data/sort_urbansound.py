import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# UrbanSound8Kの元のデータセットディレクトリ
Urban8Kdir = "/workspace/app/UrbanSound8K/audio"
# 分類後のデータセットの保存先ディレクトリ
output_dir = "/workspace/app/FSSE/data/source/urban"

# ノイズタイプに対応するディレクトリを作成
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

# 各ノイズタイプ用のディレクトリを作成
for noise_type, noise_name in noise_class_dictionary.items():
    noise_dir = os.path.join(output_dir, f"{noise_type}_{noise_name}")
    os.makedirs(noise_dir, exist_ok=True)

# UrbanSound8Kの各foldディレクトリを処理
fold_names = [f"fold{i}" for i in range(1, 11)]


# ファイルをノイズタイプごとに分類してコピーする関数
def classify_and_copy_file(file, fold_dir):
    try:
        file_parts = file.split("-")
        if len(file_parts) > 1:
            noise_type = int(file_parts[1])  # ファイル名の第2フィールドがノイズタイプ
            noise_name = noise_class_dictionary.get(noise_type)

            if noise_name:
                # 対応するディレクトリにファイルをコピー
                destination_dir = os.path.join(output_dir, f"{noise_type}_{noise_name}")
                source_path = os.path.join(fold_dir, file)
                destination_path = os.path.join(destination_dir, file)
                shutil.copy(source_path, destination_path)
                return f"Copied {file} to {destination_dir}"
    except Exception as e:
        return f"Error processing {file}: {e}"


def process_fold(fold):
    fold_dir = os.path.join(Urban8Kdir, fold)
    if os.path.exists(fold_dir):
        files = [f for f in os.listdir(fold_dir) if f.endswith(".wav")]
        return [(file, fold_dir) for file in files]
    return []


if __name__ == "__main__":
    # 各foldディレクトリのファイルを収集
    file_tasks = []
    for fold in fold_names:
        file_tasks.extend(process_fold(fold))

    # マルチスレッドでファイル分類とコピーを処理し、進捗バーを表示
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(classify_and_copy_file, file, fold_dir) for file, fold_dir in file_tasks]

        # tqdmで進捗バーを表示しながら結果を待機
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()  # 処理結果を取得
            if result:  # ログがあれば出力（エラーメッセージやコピー完了メッセージなど）
                print(result)

    print("ファイルの分類が完了しました。")
