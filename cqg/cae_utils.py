import torch
from parlai.core.metrics import F1Metric


def compute_fuzzy_recall(predicted_answers,
                         gt_answers,
                         conversational=False,
                         threshold=0.5,
                         mrecall_k=3):
    hit = 0
    conversed_gt_idx = []
    for a in predicted_answers:
        for idx, gt in enumerate(gt_answers):
            f1 = F1Metric.compute(a, gt)
            if f1.value() > threshold:
                if conversational:
                    return 1, 1
                
                if idx in conversed_gt_idx:
                    continue
                
                hit += 1
                conversed_gt_idx.append(idx)
                break

    if conversational:
        return 0, 1
    
    c = 1 if hit >= mrecall_k else 0
    return c, 1


class AnswerExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, inference=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.length = len(self.dataset)
        self.inference = inference
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def pad_ids(self, arrays, padding, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [
            array + [padding] * (max_length - len(array))
            for array in arrays
        ]
        return arrays
    
    def collate_fn(self, batch):
        input_ids = torch.tensor(self.pad_ids([b.input_id for b in batch], self.tokenizer.pad_token_id))
        input_masks = input_ids.ne(self.tokenizer.pad_token_id)
        guids = [b.guid for b in batch]
        offsets = [b.offset for b in batch]
        passages = [b.passage for b in batch]
        
        if self.inference:
            start_ids = []
            end_ids = []
            answers = [b.answers for b in batch]
        else:
            start_ids = torch.tensor([b.start_position for b in batch]).unsqueeze(-1)
            end_ids = torch.tensor([b.end_position for b in batch]).unsqueeze(-1)
            answers = []
        
        return {
            "input_ids": input_ids,
            "attention_mask": input_masks,
            "start_positions": start_ids,
            "end_positions": end_ids,
            "guids": guids,
            "offsets": offsets,
            "passages": passages,
            "answers": answers
        }