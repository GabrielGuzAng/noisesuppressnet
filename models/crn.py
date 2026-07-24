# model/crn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRN(nn.Module):
    def __init__(self, lstm_hidden=1024, n_lstm_layers=2):
        super().__init__()

        # Encoder: 5 conv2d
        self.conv1 = nn.Conv2d(1,   16,  kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv2 = nn.Conv2d(16,  32,  kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv3 = nn.Conv2d(32,  64,  kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv4 = nn.Conv2d(64,  128, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(2,3), stride=(1,2), padding=(1,0))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        # LSTM unidireccional (causal por construcción)
        self.lstm = nn.LSTM(input_size=256*4, hidden_size=lstm_hidden,
                             num_layers=n_lstm_layers, batch_first=True)

        # Decoder: 5 deconv2d con skip connections (canales doblados por concat)
        self.conv5_t = nn.ConvTranspose2d(512, 128, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv4_t = nn.ConvTranspose2d(256,  64, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv3_t = nn.ConvTranspose2d(128,  32, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.conv2_t = nn.ConvTranspose2d(64,   16, kernel_size=(2,3), stride=(1,2), padding=(1,0),
                                           output_padding=(0,1))  # ← compensa 39→80
        self.conv1_t = nn.ConvTranspose2d(32,    1, kernel_size=(2,3), stride=(1,2), padding=(1,0))

        self.bn5_t = nn.BatchNorm2d(128)
        self.bn4_t = nn.BatchNorm2d(64)
        self.bn3_t = nn.BatchNorm2d(32)
        self.bn2_t = nn.BatchNorm2d(16)
        self.bn1_t = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x esperado: [B, T, 161] (mag STFT)
        # añade dim de canal: [B, 1, T, 161]
        out = x.unsqueeze(dim=1)

        # === Encoder con truncamiento causal ===
        e1 = self.elu(self.bn1(self.conv1(out)[:, :, :-1, :].contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1)[:, :, :-1, :].contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2)[:, :, :-1, :].contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3)[:, :, :-1, :].contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4)[:, :, :-1, :].contiguous()))

        # === LSTM bottleneck ===
        # e5: [B, 256, T, 4] → [B, T, 256, 4] → [B, T, 1024]
        out = e5.transpose(1, 2).contiguous()
        q1, q2 = out.size(2), out.size(3)
        out = out.view(out.size(0), out.size(1), -1)
        out, _ = self.lstm(out)
        out = out.view(out.size(0), out.size(1), q1, q2).transpose(1, 2).contiguous()

        # Skip connection: concat con la salida del último encoder
        out = torch.cat([out, e5], dim=1)

        # === Decoder con padding causal asimétrico ===
        d5 = self.elu(torch.cat([self.bn5_t(F.pad(self.conv5_t(out), [0,0,1,0])), e4], dim=1))
        d4 = self.elu(torch.cat([self.bn4_t(F.pad(self.conv4_t(d5),  [0,0,1,0])), e3], dim=1))
        d3 = self.elu(torch.cat([self.bn3_t(F.pad(self.conv3_t(d4),  [0,0,1,0])), e2], dim=1))
        d2 = self.elu(torch.cat([self.bn2_t(F.pad(self.conv2_t(d3),  [0,0,1,0])), e1], dim=1))
        d1 = self.softplus(self.bn1_t(F.pad(self.conv1_t(d2),         [0,0,1,0])))

        # squeeze canal: [B, 1, T, 161] → [B, T, 161]
        out = torch.squeeze(d1, dim=1)
        return out
