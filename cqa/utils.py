## Most of codes are from following links
## - Transformers from Huggingface: https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/squad.py


import torch

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _find_second_max(sep_token_indices, batch_size):
    question_end_indices = []
    cur_batch, cur_max, cur_sec_max = 0, 0, 0

    for idx_batch, end_idx in to_list(sep_token_indices):
        if cur_batch != idx_batch:
            question_end_indices += [cur_sec_max]
            cur_max, cur_sec_max = 0, 0
            cur_batch = idx_batch

        if cur_max == 0:
            cur_max = end_idx
        else:
            cur_sec_max = cur_max
            cur_max = end_idx
    
    question_end_indices += [cur_sec_max]
    assert len(question_end_indices) == batch_size
        
    return question_end_indices


def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """

    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]

    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    # assert (
    #     sep_token_indices.shape[0] == 3 * batch_size
    # ), f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
    # return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]

    return _find_second_max(sep_token_indices, batch_size)


def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    question_end_index = torch.tensor(_get_question_end_index(input_ids, sep_token_id)).to(input_ids.device)
    question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
    # bool attention mask with True in locations of global attention
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.uint8)
    else:
        # last token is separation token and should not be counted and in the middle are two separation tokens
        attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(torch.uint8) * (
            attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(torch.uint8)

    return attention_mask