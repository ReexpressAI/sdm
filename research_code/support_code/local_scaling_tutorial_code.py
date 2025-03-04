# Copyright Reexpress AI, Inc. All rights reserved.

import torch
import numpy as np

# Tip: Drop this into a console and try some values to gain intuition on the re-scaling (as, for example,
# compared to softmax):

def local_soft_sdm_max(batch_input, q, distance_quantile_per_class=None, log=False, change_of_base=True):
    assert len(batch_input.shape) == 2
    assert batch_input.shape[0] == q.shape[0]
    assert q.shape[1] == 1
    if distance_quantile_per_class is not None:
        assert batch_input.shape == distance_quantile_per_class.shape
    q_rescale_offset = 2
    q_factor = q_rescale_offset + q
    batch_input = batch_input - torch.amax(batch_input, dim=1, keepdim=True)  # for numerical stability
    if distance_quantile_per_class is not None:
        rescaled_distribution = q_factor ** (batch_input * distance_quantile_per_class)
    else:
        rescaled_distribution = q_factor ** batch_input
    if log:  # log_base{q}
        kEPS = torch.finfo(torch.float32).eps  # adjust as applicable for platform
        rescaled_distribution = torch.log(rescaled_distribution + kEPS) - torch.log(
            torch.sum(rescaled_distribution, dim=1) + kEPS).unsqueeze(1)
        if change_of_base:
            return rescaled_distribution / torch.log(q_factor)
        else:
            return rescaled_distribution
    else:
        return rescaled_distribution / torch.sum(rescaled_distribution, dim=1).unsqueeze(1)


# Tip: As a reminder, when converting from log space with q as the base, we need to take the exponent
# with q as the base to recover the probability space. In our case, we also need to take into account the
# rescale offset, for which we provide a convenience function in sdm_model. Here's a local version:

def local_soft_sdm_max_log_to_probability(batch_input, q):
    assert len(batch_input.shape) == 2
    assert batch_input.shape[0] == q.shape[0]
    assert q.shape[1] == 1
    q_rescale_offset = 2
    q_factor = q_rescale_offset + q
    return q_factor ** batch_input
