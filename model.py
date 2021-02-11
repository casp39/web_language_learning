import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=32,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, feat):
        feat = feat.permute(0,2,1)
        _, (h, c) = self.lstm(feat)
        h = torch.cat([h[0], h[1]], dim=1)
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class LanguageClassificationModel(nn.Module):
    def __init__(self, base_model_path, use_feat):
        super(LanguageClassificationModel, self).__init__()
        cp = torch.load(base_model_path)
        self.wav2vec = Wav2VecModel.build_model(cp['cfg'].model, task=None)
        self.wav2vec.load_state_dict(cp['model'])
        self.use_feat = use_feat
        self.classifier = LSTM()

    def forward(self, x):
        feat = self.wav2vec.feature_extractor(x)
        output = self.classifier(feat)
        return output
