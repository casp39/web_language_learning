import sys
# sys.path.append('fairseq')
print(sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model


class SelfAttention(nn.Module):
    def __init__(self, lstm_dim, attn_h1_dim):
        super(SelfAttention, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(lstm_dim*2, attn_h1_dim),
            nn.Tanh(),
            nn.Linear(attn_h1_dim, 1)
            )
    def forward(self, out):
        return F.softmax(self.linear(out), dim=1)


class SelfAttentionClassifier(nn.Module):
  def __init__(self, lstm_dim, attn_h1_dim, attn_h2_dim, dropout_p):
    super(SelfAttentionClassifier, self).__init__()
    self.attn = SelfAttention(lstm_dim, attn_h1_dim)
    self.fc1 = nn.Linear(lstm_dim*2, attn_h2_dim)
    self.fc2 = nn.Linear(attn_h2_dim, 2)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, out):
    attention_weight = self.attn(out)
    feat = (out * attention_weight).sum(dim=1)
    x = F.relu(self.fc1(feat))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    output = F.log_softmax(x, dim=1)
    return output, attention_weight


class BiLSTM(nn.Module):
    def __init__(self, lstm_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=lstm_dim,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)

    def forward(self, feat):
        feat = feat.permute(0, 2, 1)
        output, (h, c) = self.lstm(feat)
        return output


class MingleBiLSTM(nn.Module):
    def __init__(self, lstm_dim, mingle_size):
        super(MingleBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=512*mingle_size,
                            hidden_size=lstm_dim,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)
        self.mingle_size = mingle_size

    def forward(self, feat):
        batch_size = feat.shape[0]
        seq_size = (feat.shape[2]//self.mingle_size)
        feat = feat.permute(0, 2, 1)
        feat = feat[:,:seq_size*self.mingle_size,:].reshape(batch_size, seq_size, 512*self.mingle_size)
        output, (h, c) = self.lstm(feat)
        return output


class LanguageClassificationModel(nn.Module):
    def __init__(self,
                 base_model_name,
                 base_model_path,
                 use_feat,
                 mingle_size,
                 lstm_dim,
                 attn_h1_dim,
                 attn_h2_dim,
                 dropout_p,
                 input_drop):
        super(LanguageClassificationModel, self).__init__()
        cp = torch.load(base_model_path)
        self.base_model_name = base_model_name
        self.wav2vec = Wav2Vec2Model.build_model(cp['cfg'].model, task=None)
        self.wav2vec.load_state_dict(cp['model'])
        self.use_feat = use_feat
        if not mingle_size:
            self.bilstm = BiLSTM(lstm_dim)
        else:
            self.bilstm = MingleBiLSTM(lstm_dim, mingle_size)
        self.classifier = SelfAttentionClassifier(lstm_dim, attn_h1_dim, attn_h2_dim, dropout_p)
        self.dropout = nn.Dropout(input_drop)

    def forward(self, x):
        feat = self.wav2vec.feature_extractor(x)
        output = self.bilstm(feat)
        output = self.classifier(output)
        return output
