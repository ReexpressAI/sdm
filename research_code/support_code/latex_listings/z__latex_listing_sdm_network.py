# Copyright Reexpress AI, Inc. All rights reserved.

# For listing in LaTeX.
# def pseudo_code_train():
# pdist = nn.PairwiseDistance(p=2)
# criterion = nn.NLLLoss()
# for e in range(total_epochs):
#     total_mini_batches = len(range(0, train_size, batch_size))
#     beta = min_beta
#     beta_step = (max_beta-min_beta) / total_mini_batches
#     for i in range(0, train_size, batch_size):
#         optimizer.zero_grad()
#         model.train()
#         batch_genai_y = # the next-token labels with applicable index+|V| offsets
#         # the sdm activations for the negative+positive joint distribution and the concatenation of the reference
#         # distribution with itself use the same q and distance quantiles for the corresponding instances:
#         batch_f_genai = # log_base{q} sdm activation(negative+positive linear layers output), where + is pseudo-code for concatenation
#         batch_f_original = # log_base{q} sdm activation(reference distribution+reference distribution linear layers output)
#         with torch.no_grad():
#             top_events_k = 1
#             top_k_sort_by_largest = True
#             # "negative" refers to indexes in the first half of the concatenated distributions, [0, |V|); "positive" to the second half [|V|, |V|*2):
#             neg_original_max_half_distribution_i = torch.topk(batch_f_original[:, 0:model.gen_ai_vocab],
#                                                               top_events_k, dim=1, largest=top_k_sort_by_largest)[1]
#             pos_original_max_half_distribution_i = torch.topk(batch_f_original[:, -model.gen_ai_vocab:],
#                                                               top_events_k, dim=1, largest=top_k_sort_by_largest)[1] + model.gen_ai_vocab  # note the offset
#             negative_max_half_distribution_i = torch.topk(batch_f_genai[:, 0:model.gen_ai_vocab],
#                                                           top_events_k, dim=1, largest=top_k_sort_by_largest)[1]
#             positive_max_half_distribution_i = torch.topk(batch_f_genai[:, -model.gen_ai_vocab:],
#                                                           top_events_k, dim=1, largest=top_k_sort_by_largest)[1] + model.gen_ai_vocab  # note the offset
#             distribution_mass_mask = (
#                     torch.ones_like(batch_f_genai).scatter_(1, neg_original_max_half_distribution_i, 0.0) *
#                     torch.ones_like(batch_f_genai).scatter_(1, pos_original_max_half_distribution_i, 0.0) *
#                     torch.ones_like(batch_f_genai).scatter_(1, negative_max_half_distribution_i, 0.0) *
#                     torch.ones_like(batch_f_genai).scatter_(1, positive_max_half_distribution_i, 0.0) *
#                     torch.ones_like(batch_f_genai).scatter_(1, batch_genai_y.unsqueeze(1), 0.0)
#             ).to(batch_f_genai.device)
#         regularization_term = pdist(
#             distribution_mass_mask * batch_f_original,
#             distribution_mass_mask * batch_f_genai).mean()
#         llm_loss = criterion(batch_f_genai, batch_genai_y)
#         with torch.no_grad():  # rescaling factor for the regularization term
#             regularization_scale_term = (torch.log(llm_loss + model.kEPS) /
#                                          (torch.log(regularization_term + model.kEPS) + model.kEPS)
#                                          ).item()
#         loss = llm_loss + beta * torch.sqrt(
#             torch.clamp(regularization_term, min=1.0) ** min(max(regularization_scale_term, 0.0), 1.0))
#         loss.backward()
#         optimizer.step()
#         beta += beta_step
#     # before the next epoch, for each training instance, update q and distance quantiles using the sdm activation layer trained for verification

