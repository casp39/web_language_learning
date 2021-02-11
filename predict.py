import os

import torch
from torch.utils.data import Dataset

from pydub import AudioSegment
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import LanguageClassificationModel

plt.rcParams["font.size"] = 12

# setting
device = torch.device("cpu")

test_kwargs = {'batch_size': 1, 'shuffle': False}

classes = ['関西弁', '標準語']
mingle_size = 4

def read_audio(fname):
    wav, sr = sf.read(fname)
    assert sr == 16e3

    L = 4*16000
    if len(wav) < L:
        tempData = np.zeros((L))
        tempData[:len(wav)] = wav
        return tempData
    else:
        return wav

class TestDialectDataset(Dataset):
    def __init__(self, audiofile):
        self.audiofile=audiofile

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.audiofile

def save_graph_data(audio_file, attention_weight, pred, save_img_url):
    # save original wave data
    t = np.arange(0, len(audio_file))/16000
    plt.figure(figsize=(10,4))
    plt.subplots_adjust(bottom=0.2)
    plt.xlim([0, 4.0])
    plt.ylim([-1, 1])
    plt.plot(t, audio_file)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    # plt.axis("off")
    # plt.gca().clear()

    attention_weight = attention_weight/max(attention_weight)
    for i, w in enumerate(attention_weight):
        if w > 0.3:
            plt.plot(np.arange(int(16000*(0.020*i*mingle_size)), int(16000*(0.005+0.020*(i+1)*mingle_size)))/16000, 
                     audio_file[int(16000*(0.020*i*mingle_size)):int(16000*(0.005+0.020*(i+1)*mingle_size))], 
                     color='red', alpha=w)

    # save wave data with attention
    # plt.plot(t, audio_file)
    plt.savefig(save_img_url)
    plt.gca().clear()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            output, attention_weight = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            audio_file = data.reshape(-1).to('cpu').detach().numpy()
            attention_weight = attention_weight.reshape(-1).to('cpu').detach().numpy()

            pred = classes[pred.item()]

    return audio_file, attention_weight, pred


def identify(model, audiofile, save_img_url):
    audio = read_audio(audiofile)

    test_set = TestDialectDataset(audio)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    audio_file, attention_weight, pred = test(model, device, test_loader)
    save_graph_data(audio_file, attention_weight, pred, save_img_url)
    return audio_file, attention_weight, pred 

