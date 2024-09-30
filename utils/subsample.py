import torch

def subsample2(wav, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav1 = []
    wav2 = []
    for i in range(len(wav)):
        channel, length = wav[i].shape
        new_length = length // 2 - 128

        # ランダムなインデックスを生成
        indices = (torch.arange(new_length, device=device) * 2 - 127).unsqueeze(0).repeat(channel, 1)

        # インデックスの境界を処理
        indices = indices.clamp(0, length - 2)

        random_choice = torch.randint(0, 2, (channel, new_length), device=device)

        # ランダムに選択されたインデックスを作成
        index1 = indices + random_choice
        index2 = indices + (1 - random_choice)

        # 新しいテンソルを作成
        wav1.append( torch.gather(wav[i].to(device), 1, index1))
        wav2.append( torch.gather(wav[i].to(device), 1, index2))
    wav1 = torch.cat(wav1,dim=0).unsqueeze(1)
    wav2 = torch.cat(wav2,dim=0).unsqueeze(1)
    return wav1, wav2