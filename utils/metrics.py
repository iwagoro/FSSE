import torch
import torchaudio
from rich.progress import Progress
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from .stft import istft

SAMPLE_RATE = 48000


def getPesqList(cleans, preds, type):
    psq = []
    pesq_metric = PerceptualEvaluationSpeechQuality(16000, type)
    for i in range(min(len(cleans), len(preds))):
        pred = istft(preds[i])
        clean = istft(cleans[i])

        clean_resampled = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(clean.detach().cpu())
        pred_resampled = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(pred.detach().cpu())

        score = pesq_metric(pred_resampled.flatten(), clean_resampled.flatten())
        if not torch.isinf(score):
            psq.append(score)

    return sum(psq).item()


def getSNRList(cleans, preds):
    snr = []
    snr_metric = SignalNoiseRatio()

    for i in range(min(len(cleans), len(preds))):
        pred = istft(preds[i])
        clean = istft(cleans[i])

        clean_resampled = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(clean.detach().cpu())
        pred_resampled = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(pred.detach().cpu())

        score = snr_metric(pred_resampled.flatten(), clean_resampled.flatten())
        if not torch.isinf(score):
            snr.append(score)

    return sum(snr).item()


def getSTOIList(cleans, preds):
    stoi = []
    stoi_metric = ShortTimeObjectiveIntelligibility(16000, False)

    for i in range(min(len(cleans), len(preds))):
        pred = istft(preds[i])
        clean = istft(cleans[i])

        clean_resampled = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(clean.detach().cpu())
        pred_resampled = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)(pred.detach().cpu())

        score = stoi_metric(pred_resampled.flatten(), clean_resampled.flatten())
        if not torch.isinf(score):
            stoi.append(score)

    return sum(stoi).item()
