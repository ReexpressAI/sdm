# Copyright Reexpress AI, Inc. All rights reserved.

# Streamlined code appearing in LaTeX listing
# def sdm_activation_function(batch_input, q, distance_quantile_per_class=None, log=False):
#     """
#     sdm activation function
#     Parameters
#     ----------
#     batch_input
#         torch.tensor
#             shape == [batch size, number of classes]
#     q
#         torch.tensor
#             shape == [batch size, 1], with each value in [0, max q]
#     distance_quantile_per_class
#         torch.tensor, or None
#             If not None, shape == [batch size, number of classes], with each quantile in [0,1]. As a final layer
#             activation function, with batch_input $\in \reals$, it is assumed that the quantiles are the same
#             across classes, for a given instance. This ensures the argmax does not change relative to
#             torch.argmax(batch_input, dim=1).
#     log
#         log with change of base, for training
#     Notes:
#         For context, with e.g. batch size = 1, the standard softmax is obtained by using q=torch.tensor([[torch.e-2]])
#         and (distance_quantile_per_class=None or distance_quantile_per_class=torch.ones(1, number of classes) ).
#     Returns
#     -------
#     [batch size, number of classes]
#     """
#     assert len(batch_input.shape) == 2
#     assert batch_input.shape[0] == q.shape[0]
#     assert q.shape[1] == 1
#     if distance_quantile_per_class is not None:
#         assert batch_input.shape == distance_quantile_per_class.shape
#     q_rescale_offset = 2
#     q_factor = q_rescale_offset + q
#     batch_input = batch_input - torch.amax(batch_input, dim=1, keepdim=True)  # for numerical stability
#     if distance_quantile_per_class is not None:
#         rescaled_distribution = q_factor ** (batch_input * distance_quantile_per_class)
#     else:
#         rescaled_distribution = q_factor ** batch_input
#     if log:  # log_base{q}
#         kEPS = torch.finfo(torch.float32).eps  # adjust as applicable for platform
#         rescaled_distribution = torch.log(rescaled_distribution + kEPS) - torch.log(
#             torch.sum(rescaled_distribution, dim=1) + kEPS).unsqueeze(1)
#         return rescaled_distribution / torch.log(q_factor)
#     else:
#         return rescaled_distribution / torch.sum(rescaled_distribution, dim=1).unsqueeze(1)

