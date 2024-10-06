import torch
from torchmetrics.regression import MeanSquaredError
from .stft import stft, istft


def tensor_stft(wav):
    result = []

    for i in range(len(wav)):
        wav_stft = stft(wav[i])
        result.append(wav_stft)

    result = torch.cat(result, dim=0).unsqueeze(1)
    return result


def stft_loss(clean, pred):
    clean_stft = tensor_stft(clean)
    pred_stft = tensor_stft(pred)

    size = clean_stft.size()

    t_length = size[2]
    f_length = size[3]
    loss = 0

    for i in range(len(clean_stft)):
        clean_r = clean_stft[i][..., 0]
        clean_i = clean_stft[i][..., 1]
        pred_r = pred_stft[i][..., 0]
        pred_i = pred_stft[i][..., 1]

        clean_diff = torch.abs(clean_r) + torch.abs(clean_i)
        pred_diff = torch.abs(pred_r) + torch.abs(pred_i)

        loss += torch.abs(clean_diff - pred_diff)

    return torch.sum(loss) / (t_length * f_length)


def mse_loss(clean, pred, device):
    mean_squared_error = MeanSquaredError().to(device)
    loss = 0
    for i in range(len(clean)):
        loss += mean_squared_error(pred[i].flatten(), clean[i].flatten())
    return loss


def wsdr_loss(x, y_pred, y_true, eps=1e-8):
    # # to time-domain waveform
    y_true = torch.squeeze(y_true, 1)
    y_pred = torch.squeeze(y_pred,1)
    x = torch.squeeze(x, 1)
    y_true = istft(y_true)
    y_pred = istft(y_pred)
    x = istft(x)

    y_pred = y_pred.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)


def basic_loss(g1, g2, fg1, device):
    alpha = 0.8
    loss_time = mse_loss(fg1, g2, device)
    loss_freq = stft_loss(fg1, g2)
    loss = (alpha * loss_time + (1 - alpha) * loss_freq) / 600 + wsdr_loss(g1, fg1, g2)
    return loss


def reg_loss(fg1, g2, gf1, gf2):
    loss = torch.mean((fg1 - g2 - (gf1 - gf2)) ** 2)
    return loss
