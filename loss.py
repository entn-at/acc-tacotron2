import torch
import torch.nn as nn

from torch import nn


class DNNLoss(nn.Module):
    """Tacotron2 Loss"""

    def __init__(self):
        super(DNNLoss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_out, mel_out_postnet, gate_out, _ = model_output

        mel_loss = nn.MSELoss()(mel_out, mel_target)
        mel_postnet_loss = nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        return mel_loss, mel_postnet_loss, gate_loss


# class DNNLoss(nn.Module):
#     def __init__(self):
#         super(DNNLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()

#     def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target):
#         mel_target.requires_grad = False
#         mel_loss = self.mse_loss(mel, mel_target)
#         mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)
#         duration_predictor_target.requires_grad = False
#         duration_predictor_loss = self.l1_loss(duration_predicted,
#                                                duration_predictor_target.float())

#         return mel_loss, mel_postnet_loss, duration_predictor_loss
