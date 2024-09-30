import torch

N_FFT = 1022
HOP_LENGTH = 256


#! １つの波形に対してstftを行う関数
def stft(wav):
    window = torch.hann_window(N_FFT, device=wav.device)
    stft = torch.stft(
        input=wav, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, return_complex=True, window=window
    )
    stft = torch.view_as_real(stft)
    return stft


#! 1つのstftに対してistftを行う関数
def istft(stft):
    window = torch.hann_window(N_FFT, device=stft.device)
    stft = torch.view_as_complex(stft)
    wav = torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=window)
    return wav


#! バッチの波形に対してstftを行う関数
def tensor_stft(wav):
    result = []

    for i in range(len(wav)):
        wav_stft = stft(wav[i])
        result.append(wav_stft)

    result = torch.cat(result, dim=0).unsqueeze(1)
    return result


#! バッチのstftに対してistftを行う関数
def tensor_istft(stft):
    result = []

    for i in range(len(stft)):

        wav = istft(stft[i])
        result.append(wav)

    result = torch.cat(result, dim=0).unsqueeze(1)
    return result
