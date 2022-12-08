import json
import torch
from thefuzz import fuzz
import random
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from .data_utils import AnswerSpanAligner

HL_TOKEN = '<hl>'
MAX_SEQ_LENGTH = 512

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


@dataclass
class ISCQGInstance:
    title: str
    section_title: str
    background: str
    context: str
    history: List[Union[None, str]]
    qas: List[Union[None, str]]
    guid: Optional[str] = None
    max_history_length: Optional[int] = 32
    task: Optional[str] = "quac"
        
    def set_aligner(self, tokenizer, get_offset=False, add_cannot_answer=False):
        if add_cannot_answer:
            c = len(self.context)
            cannot_answer = {
                 'score': 0.,
                 'text': 'CANNOTANSWER',
                 'start': c + 1,
                 'end': c + 12
            }
            # For CoQA
            yesno_answer = {
                'score': 0.,
                'text': 'YESNO',
                'start': c + 14,
                'end': c + 18
            }
            self.cannot_answer = cannot_answer
            self.yesno_answer = yesno_answer
            if self.task == "quac":
                context = self.context + ' CANNOTANSWER'
            else:  # coqa
                context = self.context + ' CANNOTANSWER YESNO'
        else:
            self.cannot_answer = None
            self.yesno_answer = None
            context = self.context

        self.aligner = AnswerSpanAligner(context, tokenizer, get_offset=get_offset)

    def make_cqg_input(self, tokenizer):
        assert len(self.history) % 2 == 0
        assert len(self.priors)

        a = self.priors.pop(0)

        passage = self.aligner.get_highlight_by_answer(tokenizer,
                                                       a.get('start') or a.get('answer_start', 0),
                                                       a['text'],
                                                       384,
                                                       HL_TOKEN
                                                      )
        passage = tokenizer.convert_tokens_to_ids(passage)
        available_length = MAX_SEQ_LENGTH - len(passage) - 2
        
        if self.task == "coqa" and a["text"] in ["CANNOTANSWER", "YESNO"]:
            a["start"] = -1
            if a["text"] == "CANNOTANSWER":
                a["text"] = "unknown"
            else:
                a["text"] = "yes" if random.random() < 0.5 else "no"
        
        context = self.history + ['<extra_id_0>', a['text']]
        context = f" {tokenizer.eos_token} ".join(context)
        context = tokenizer.encode(context, add_special_tokens=False)

        if len(context) > available_length:
            gap = len(context) - available_length
            context = context[gap:]
        
        input_id = passage + [tokenizer.eos_token_id] + context + [tokenizer.eos_token_id]
        return torch.tensor([input_id]), a['text'], a['start']
    
    def make_ae_input(self, tokenizer):
        guid = None
        available_length = MAX_SEQ_LENGTH - 2
        passage = self.aligner.all_doc_tokens
        passage = tokenizer.convert_tokens_to_ids(passage)
        passage = passage[:available_length]
        input_id = [tokenizer.cls_token_id] + passage + [tokenizer.sep_token_id]
        offset = self.aligner.offset
        instance = AnswerExtractionExample(guid,
                                           self.context,
                                           input_id,
                                           offset,
                                           [])
        return instance
    
    def set_answer_priors(self, priors,
                          shuffle=False,
                          dedup_ratio=50,
                          add_cannot_answer=False,
                          max_cannot_answer=3):
        priors = self.dedup_answers(priors, dedup_ratio, add_cannot_answer, max_cannot_answer)
        if shuffle:
            random.shuffle(priors)
        self.priors = priors

    def dedup_answers(self, answers, ratio=50, add_cannot_answer=False, max_cannot_answer=3):
        answers = sorted(answers, key=lambda x: len(x['text']))
        skip_indices = []
        for i, answer in enumerate(answers):
            if answer['text'].strip() == '':
                skip_indices.append(i)
                continue
                
            if answer['start'] > answer['end']:
                skip_indices.append(i)
                continue
            
            for j in range(len(answers)):
                if j in skip_indices:
                    continue

                if i == j:
                    continue

                s = fuzz.ratio(answer['text'], answers[j]['text'])
                if s > ratio or answer['text'] in answers[j]['text']:
                    skip_indices.append(i)
                    break

        dedupped_answers = []
        n_cannot_answer = 0
        for i, a in enumerate(answers):
            if i in skip_indices:
                if add_cannot_answer and self.cannot_answer and n_cannot_answer < max_cannot_answer:
                    self.cannot_answer['score'] = a['score']
                    if self.task == "coqa" and random.random() < 0.5:
                        dedupped_answers.append(self.yesno_answer)
                    else:
                        dedupped_answers.append(self.cannot_answer)
                    n_cannot_answer += 1
                continue
            dedupped_answers.append(a)
        return sorted(dedupped_answers, key=lambda x: x['score'], reverse=True)
    
    def add_context(self, text):
        self.history.append(text)

    def add_qa(self, qa):
        self.qas.append(qa)
        
    def save(self, output_path):
        json.dump(dataclasses.asdict(self),
                  open(output_path, 'w', encoding='utf-8'),
                  ensure_ascii=False)

    def save_to_jsonl(self, output_path):
        with open(output_path, 'a', encoding='utf-8') as outfile:
            json.dump(dataclasses.asdict(self),
                        outfile,
                        ensure_ascii=False)
            outfile.write('\n')

        
    def is_done(self):
        if not hasattr(self, 'priors'):
            return False

        if self.priors:
            return False

        return True

    
@dataclass
class ISSequentialCQGInstance:
    title: str
    section_title: str
    background: str
    context: str
    history: List[Union[None, str]]
    qas: List[Union[None, str]]
    guid: Optional[str] = None
    max_history_length: Optional[int] = 32
    task: Optional[str] = "quac"
        
    def set_aligner(self, cae_tokenizer, cqg_tokenizer):
        c = len(self.context)
        cannot_answer = {
             'score': 0.,
             'text': 'CANNOTANSWER',
             'start': c + 1,
             'end': c + 13
        }
        yesno_answer = {
                'score': 0.,
                'text': 'YESNO',
                'start': c + 14,
                'end': c + 18
            }
        self.cannot_answer = cannot_answer
        self.yesno_answer = yesno_answer
        if self.task == "quac":
            context = self.context + ' CANNOTANSWER'
        else:  # coqa
            context = self.context + ' CANNOTANSWER YESNO'
        
        self.cae_aligner = AnswerSpanAligner(self.context, cae_tokenizer, get_offset=True)
        self.cqg_aligner = AnswerSpanAligner(context, cqg_tokenizer, get_offset=False)

    def make_cqg_input(self, tokenizer):
        assert len(self.history) % 2 == 0
        assert len(self.priors)

        a = self.priors.pop(0)
        assert "start" in a or "answer_start" in a
        passage = self.cqg_aligner.get_highlight_by_answer(tokenizer,
                                                           a.get('start') or a.get('answer_start', 0),
                                                           a['text'],
                                                           384,
                                                           HL_TOKEN
                                                          )
        passage = tokenizer.convert_tokens_to_ids(passage)
        available_length = MAX_SEQ_LENGTH - len(passage) - 2
        
        if self.task == "coqa" and a["text"] in ["CANNOTANSWER", "YESNO"]:
            a["start"] = -1
            if a["text"] == "CANNOTANSWER":
                a["text"] = "unknown"
            else:
                a["text"] = "yes" if random.random() < 0.5 else "no"
        
        context = self.history + ['<extra_id_0>', a['text']]
        context = f" {tokenizer.eos_token} ".join(context)
        context = tokenizer.encode(context, add_special_tokens=False)

        if len(context) > available_length:
            gap = len(context) - available_length
            context = context[gap:]
        
        input_id = passage + [tokenizer.eos_token_id] + context + [tokenizer.eos_token_id]
        return torch.tensor([input_id]), a['text'], a['start']
    
    def make_cae_input(self, tokenizer):
        guid = None
        available_length = MAX_SEQ_LENGTH - 2 - self.max_history_length
        passage = self.cae_aligner.all_doc_tokens
        passage = tokenizer.convert_tokens_to_ids(passage)
        passage = passage[:available_length]
        input_id = [tokenizer.cls_token_id] + passage + [tokenizer.sep_token_id]
        offset = self.cae_aligner.offset
        
        history = f" {tokenizer.sep_token} ".join(self.history[-2:]) # user last 2 turns
        history_id = tokenizer.encode(history, add_special_tokens=False)
        if len(history_id) > (self.max_history_length - 1):
            gap = len(history_id) - (self.max_history_length - 1)
            history_id = history_id[gap:]
        history_id.append(tokenizer.sep_token_id)

        input_id = input_id + history_id
        instance = AnswerExtractionExample(guid,
                                           self.context,
                                           input_id,
                                           offset,
                                           [])
        return instance
    
    def set_answer_priors(self, priors,
                          shuffle=False,
                          dedup_ratio=50,
                          add_cannot_answer=False,
                          max_cannot_answer=3):
        if dedup_ratio:
            priors = self.dedup_answers(priors, dedup_ratio, add_cannot_answer, max_cannot_answer)
        if shuffle:
            random.shuffle(priors)
        self.priors = priors

    def dedup_answers(self, answers, ratio=50, add_cannot_answer=False, max_cannot_answer=3):
        
        previous_answers = [a for i, a in enumerate(self.history) if i % 2 == 1]
        answers = sorted(answers, key=lambda x: len(x['text']))
        skip_indices = []
        for i, answer in enumerate(answers):
            if answer['text'].strip() == '':
                skip_indices.append(i)
                continue
                
            if answer['start'] > answer['end']:  #  or answer['start'] == 0
                skip_indices.append(i)
                continue
            
            for j in range(len(answers)):
                if j in skip_indices:
                    continue

                if i == j:
                    continue

                s = fuzz.ratio(answer['text'], answers[j]['text'])
                if s > 50 or answer['text'] in answers[j]['text']:
                    skip_indices.append(i)
                    break
            
            # check overlap with previous answers
            for pa in previous_answers:
                s = fuzz.ratio(answer['text'], pa)
                if (s > ratio or answer['text'] in pa) and i not in skip_indices:
                    skip_indices.append(i)
                    break

        dedupped_answers = []
        n_cannot_answer = 0
        for i, a in enumerate(answers):
            if i in skip_indices:
                if add_cannot_answer and self.cannot_answer and n_cannot_answer < max_cannot_answer:
                    if self.task == "coqa" and random.random() < 0.5:
                        dedupped_answers.append(self.yesno_answer)
                    else:
                        dedupped_answers.append(self.cannot_answer)
                    n_cannot_answer += 1
                continue
            dedupped_answers.append(a)
        return sorted(dedupped_answers, key=lambda x: x['score'], reverse=True)
    
    def add_context(self, text):
        self.history.append(text)

    def add_qa(self, qa):
        self.qas.append(qa)
        
    def save(self, output_path):
        json.dump(dataclasses.asdict(self),
                  open(output_path, 'w', encoding='utf-8'),
                  ensure_ascii=False)

    def save_to_jsonl(self, output_path):
        with open(output_path, 'a', encoding='utf-8') as outfile:
            json.dump(dataclasses.asdict(self),
                        outfile,
                        ensure_ascii=False)
            outfile.write('\n')
