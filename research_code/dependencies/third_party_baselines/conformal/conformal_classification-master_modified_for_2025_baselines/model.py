# This simply serves as a wrapper to pass through the pre-computed logits. We have structured this in this way
# in order to minimize changes to the existing baseline code and to make it easy to experiment with additional
# transformations by just modifying forward().

import torch
import torch.nn as nn
import torch.nn.functional as F


class Passthrough(nn.Module):
    def __init__(self, **kwargs):
        super(Passthrough, self).__init__()

        self.class_size = kwargs["class_size"]

    def forward(self, pre_computed_logits):
        # return torch.softmax(pre_computed_logits, dim=1)
        return pre_computed_logits