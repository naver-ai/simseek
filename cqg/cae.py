import json
from tqdm import tqdm
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .span import AnswerSpanAligner
from .sae import AnswerExtractionExample, load_processed_data, load_dataset


def load_processed_data(data_path):
    examples = []
    for line in open(data_path, "r", encoding="utf-8"):
        examples.append(json.loads(line))
    return examples


def load_dataset(data_path, tokenizer, max_seq_length=512, hl_token="<hl>"):
    data = json.load(open(data_path, "r"))
    examples = []
    for d in tqdm(data["data"]):
        example = make_instances(d, tokenizer, max_seq_length, hl_token)
        examples.extend(example)
    return examples


@dataclass
class AnswerExtractionExample:
    guid: str
    passage: str
    input_id: List[int]
    offset: List[Tuple[int, int]]
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    answers: Optional[List[dict]] = None
        
    def to_dict(self):
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)


def make_instances(data,
                   tokenizer,
                   max_seq_length=512,
                   max_history_length=128,
                   use_previous_qa=False,
                   use_previous_answer=False,
                   hl_token="<hl>",
                   skip_cannotanswer=True,
                   inference=False):
    instances = []
    for paragraph in data["paragraphs"]:
        passage_str = paragraph["context"]
        if skip_cannotanswer:
            passage_str = passage_str.replace("CANNOTANSWER", "").rstrip()
        
        aligner = AnswerSpanAligner(passage_str, tokenizer, get_offset=True)
        guid = None
        
        available_length = max_seq_length - 2 - max_history_length
        passage = aligner.all_doc_tokens
        passage = tokenizer.convert_tokens_to_ids(passage)
        passage = passage[:available_length]
        input_ = [tokenizer.cls_token_id] + passage + [tokenizer.sep_token_id]
        offset = aligner.offset
        
        conv = []
        for tid, qas in enumerate(paragraph["qas"]):
            a = qas["orig_answer"]
            guid = qas["id"]
            
            if a["text"] == "CANNOTANSWER" and skip_cannotanswer:
                continue

            start, end = aligner.get_span_in_subwords(a["answer_start"],
                                                      a["text"],
                                                      tokenizer)

            # FOR CLS token
            start += 1
            end += 1
            if (start >= available_length or end >= available_length) and not inference:
                continue

            turn_guid = f"{guid}"
            if use_previous_qa:
                history = f" {tokenizer.sep_token} ".join(conv[-2:])
            else:
                history = f" {tokenizer.sep_token} ".join(conv)
            history_id = tokenizer.encode(history, add_special_tokens=False)
            if len(history_id) > (max_history_length - 1):
                gap = len(history_id) - (max_history_length - 1)
                history_id = history_id[gap:]
            history_id.append(tokenizer.sep_token_id)

            input_id = input_ + history_id

            instance = AnswerExtractionExample(turn_guid,
                                               passage_str,
                                               input_id,
                                               offset,
                                               start,
                                               end,
                                               answers=[qas])
            
            
            instances.append(instance)

            conv.append(qas["question"])
            conv.append(qas["orig_answer"]["text"])
    return instances
