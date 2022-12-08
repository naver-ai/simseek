from dataclasses import dataclass
from typing import List

import json
from copy import deepcopy
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification

import argparse
from .archive.eval import normalize_answer
import string
from difflib import SequenceMatcher


def post_processing_question(question):
    question = question.strip("VE:")
    question = question.lstrip(string.punctuation)
    return question.strip()


def overlap_fiter(previous, target, ratio=0.9):    
    scores = []
    for p in previous:
        s = SequenceMatcher(None, target, p).ratio()
        scores.append(s)

    if max(scores) > ratio:
        return True
    return False


def process_data(data, vtype, tokenizer):
    instances = []
    questions = []
    for i in range(len(data['data'])):
        title = data['data'][i]['title']
        subtitle = data['data'][i]['section_title']
        background = data['data'][i]['background']
        for p in data['data'][i]['paragraphs']:
            context = p['context']
            history = []
            for qa in p['qas']:
                guid = qa['id']
                question = qa['question']
                answer = qa['orig_answer']['text']

                instance = VerifyData(guid=guid,
                                      history=deepcopy(history),
                                      question=question,
                                      answer=answer,
                                      title=title,
                                      subtitle=subtitle,
                                      background=background,
                                      context=context)
                if vtype == "question":
                    instance.make_question_base(tokenizer)
                else:
                    instance.make_answer_base(tokenizer)
                instances.append(instance)

                history.append(question)
                history.append(answer)
                questions.append(normalize_answer(question))
        if i % 100 == 0 or i == len(data['data']) - 1:
            print(f"{i}/{len(data['data'])}", end="\r")
            
    frequent_q = [k for k, v in Counter(questions).most_common() if v > 1]
    context_to_id = defaultdict(list)
    for idx, instance in enumerate(instances):
        cid = instance.guid.split("#")[0]
        context_to_id[cid].append(idx)
    
    return instances, frequent_q, context_to_id


@dataclass
class VerifyData:
    guid: str
    history: List[str]
    question: str
    answer: str
    title: str
    subtitle: str
    background: str
    context: str
        
    def make_answer_base(self, tokenizer, max_length=384):
        context_inputs = tokenizer.encode(self.context, add_special_tokens=False)
        a = tokenizer.encode(self.answer, add_special_tokens=False)
        a_id_string = " ".join([str(aa) for aa in a])
        c_id_string = " ".join([str(cc) for cc in context_inputs[:max_length]])
        if a_id_string in c_id_string or self.answer == "CANNOTANSWER":
            self.context_inputs = context_inputs[:max_length]
        else:
            self.context_inputs = context_inputs[-max_length:]
    
    def make_question_base(self, tokenizer, max_length=384):
        k = f" {tokenizer.sep_token} ".join([self.title, self.subtitle, self.background])
        context_inputs = tokenizer.encode(k, add_special_tokens=False)
        context_inputs = context_inputs[:max_length]
        self.context_inputs = context_inputs
        
    def make_data_for_answer(self, tokenizer, negative=None):
        h = f" {tokenizer.sep_token} ".join(self.history + [self.question])

        if negative:
            target = negative
            label = 0
        else:
            target = self.answer
            label = 1

        history_inputs = tokenizer.encode(h, add_special_tokens=False)
        target_inputs = tokenizer.encode(target, add_special_tokens=False)
        inputs = history_inputs + [tokenizer.sep_token_id] + target_inputs
        
        max_available_length = 512 - len(self.context_inputs) - 3
        if len(inputs) > max_available_length:
            gap = len(inputs) - max_available_length
            inputs = inputs[gap:]
            
        inputs = [tokenizer.cls_token_id] + self.context_inputs + [tokenizer.sep_token_id] + inputs + [tokenizer.sep_token_id]        
        token_type = [0] * (len(inputs) - (len(target_inputs) + 1))
        token_type.extend([1] * (len(target_inputs) + 1))
        return inputs, token_type, label

    def make_data_for_question(self, tokenizer, negative=None):
        h = f" {tokenizer.sep_token} ".join(self.history)

        if negative:
            target = negative
            label = 0
        else:
            target = self.question
            label = 1

        history_inputs = tokenizer.encode(h, add_special_tokens=False)
        target_inputs = tokenizer.encode(target, add_special_tokens=False)
        inputs = history_inputs + [tokenizer.sep_token_id] + target_inputs
        
        max_available_length = 512 - len(self.context_inputs) - 3
        if len(inputs) > max_available_length:
            gap = len(inputs) - max_available_length
            inputs = inputs[gap:]
            
        inputs = [tokenizer.cls_token_id] + self.context_inputs + [tokenizer.sep_token_id] + inputs + [tokenizer.sep_token_id]
        
        token_type = [0] * (len(inputs) - (len(target_inputs) + 1))
        token_type.extend([1] * (len(target_inputs) + 1))
        return inputs, token_type, label


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]
    return arrays


class VerifyDataset(Dataset):
    def __init__(self,
                 verifying_target,
                 dataset,
                 tokenizer,
                 rng,
                 context_to_id={},
                 frequent_q=[],
                 inference=False):
        assert verifying_target in ['question', 'answer']
        self.verifying_target = verifying_target
        self.dataset = dataset
        
        if self.verifying_target == 'answer' and not inference:
            assert context_to_id
        
        if self.verifying_target == 'question' and not inference:
            assert frequent_q
        
        self.context_to_id = context_to_id
        self.frequent_q = frequent_q
        self.tokenizer = tokenizer
        self.rng = rng
        self.inference = inference
        
    def sample_negative_question(self):
        i = self.rng.choice(self.dataset)
        return i.question
    
    def _prepare_answer(self, instance, idx):
        if self.inference:
            input_ids, token_type_ids, label = instance.make_data_for_answer(self.tokenizer)
            return instance.guid, input_ids, token_type_ids, label
        
        cid = instance.guid.split("#")[0]
        if self.rng.random() < 0.5:
            nid = idx
            while nid == idx:
                nid = self.rng.choice(self.context_to_id[cid])
            negative = self.dataset[nid].answer
        else:
            negative = None
        
        input_ids, token_type_ids, label = instance.make_data_for_answer(self.tokenizer, negative)
        return instance.guid, input_ids, token_type_ids, label
    
    def _prepare_question(self, instance):
        if self.inference:
            input_ids, token_type_ids, label = instance.make_data_for_question(self.tokenizer)
            return instance.guid, input_ids, token_type_ids, label
        
        if normalize_answer(instance.question) in self.frequent_q:
            negative = instance.question
        elif self.rng.random() < 0.5:
            if self.rng.random() < 0.5:
                negative = random.choice(self.frequent_q) + "?"
            else:
                negative = self.sample_negative_question()
        else:
            negative = None
        
        input_ids, token_type_ids, label = instance.make_data_for_question(self.tokenizer, negative)
        return instance.guid, input_ids, token_type_ids, label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.verifying_target == 'question':
            return self._prepare_question(self.dataset[idx])
        else:
            return self._prepare_answer(self.dataset[idx], idx)
    
    def collate_fn(self, batch):
        guids, input_ids, token_type_ids, labels = list(zip(*batch))
        input_ids = pad_ids(input_ids, self.tokenizer.pad_token_id)
        input_ids = torch.tensor(input_ids)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        token_type_ids = pad_ids(token_type_ids, 0)
        token_type_ids = torch.tensor(token_type_ids)
        labels = torch.tensor(labels)
        
        return input_ids, token_type_ids, attention_mask, labels, guids
