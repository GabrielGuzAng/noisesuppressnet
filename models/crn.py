# model/crn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRN(nn.Module):
    """Causal CRN (Tan & Wang 2018) con compuerta opcional de paso directo.

    Args:
        lstm_hidden: hidden size of the bottleneck LSTM.
        n_lstm_layers: number of LSTM layers.
        gate: if True, adds a causal per-band pass-through gate. The output
            becomes ``g * M_hat + (1 - g) * M_noisy`` with ``g`` in [0, 1]
            predicted from the same decoder features that feed ``conv1_t``.

            The baseline architecture does pure spectral mapping: it has no
            cheap way to express "leave this frame alone", because not
            modifying a bin requires reconstructing it exactly from the LSTM
            bottleneck. The gate adds that identity path at ~193 parameters.
            Its successor GCRN (Tan & Wang 2020) does not have one either: its
            gates are GLUs between conv layers, not a path to the input.

            With ``gate=False`` this class is bit-identical to the original.
    """

    def __init__(self, lstm_hidden=1024, n_lstm_layers=2, gate=False):
        super().__init__()
        self.gate = gate

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

        if gate:
            # Mirrors conv1_t: same input (d2, 32 channels), same kernel and
            # stride, so the gate inherits the decoder's causal structure.
            # No BatchNorm here on purpose: it would fight the bias init below.
            self.conv_gate = nn.ConvTranspose2d(32, 1, kernel_size=(2, 3),
                                                stride=(1, 2), padding=(1, 0))
            # Null-preserving init: g0 ~= 0.953, so the gated model starts
            # essentially at the ungated one and can only learn to back off if
            # the loss rewards it. std=0.01 keeps g input-dependent from step 1
            # while staying in a region where the sigmoid still has gradient.
            nn.init.normal_(self.conv_gate.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.conv_gate.bias, 3.0)

    def forward(self, x, return_gate=False):
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

        if self.gate:
            # The transposed conv yields T-1 frames; frame 0 has no decoder
            # features behind it. Padding the LOGITS with a zero would force
            # g = 0.5 there (a 50/50 blend on the first frame regardless of
            # what the model learned), so the pad carries the learned bias
            # instead: with no evidence, fall back to the default gate value.
            logits = self.conv_gate(d2)
            pad = self.conv_gate.bias.view(1, 1, 1, 1).expand(
                logits.size(0), 1, 1, logits.size(3))
            g = torch.sigmoid(torch.cat([pad, logits], dim=2))
            g = torch.squeeze(g, dim=1)
            out = g * out + (1.0 - g) * x
            if return_gate:
                return out, g

        if return_gate:
            return out, None
        return out
