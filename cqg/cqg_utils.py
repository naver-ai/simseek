import json
import torch
import random
from tqdm import tqdm
from .span import AnswerSpanAligner, SpanMapper


HL_TOKEN = "<hl>"
SPECIAL_TOKENS = ["<hl>"]


class CQGDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.length = len(self.dataset)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, batch):
        inputs, targets, guids = list(zip(*batch))
        input_ids = torch.tensor(pad_ids(inputs, self.tokenizer.pad_token_id))
        target_ids = torch.tensor(pad_ids(targets, self.tokenizer.pad_token_id))
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return input_ids, attention_mask, target_ids, list(guids)


def get_answer_containing_answers(context, answers):
    mapper = SpanMapper(context)
    new_answer = []
    for answer in answers:
        answer_start, text = mapper.align(answer['answer_start'], answer['text'])
        new_answer.append({'text': text, 'answer_start': answer_start})
    return new_answer


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]
    return arrays


def load_and_prepare_inference(data_path, tokenizer, answer_prior='span'):
    data = json.load(open(data_path, 'r'))
    examples = []
    for idx in tqdm(range(len(data['data']))):
        context = data['data'][idx]['paragraphs'][0]['context']
        answers = [qa['orig_answer'] for qa in data['data'][idx]['paragraphs'][0]['qas']] #x[did][0]
        if answer_prior == 'sent':
            answers = get_answer_containing_answers(context, answers)
#         questions = [qa['question'] for qa in data['data'][idx]['paragraphs'][0]['qas']]
        # for a in answers:
        #     a['answer_start'] = a['start']
        # answers = sorted(answers, key=lambda x: x['start'])
        did = data['data'][idx]['paragraphs'][0]['id']
        I = IterativeCQGInference(did, context, answers)
        I.tokenize_base(tokenizer)
        examples.append(I)
    return examples


class IterativeCQGInference:
    
    def __init__(self, did, context, answers, max_passage_length=384, max_seq_length=512):
        self.context = context
        self.answers = answers
        self.max_passage_length = max_passage_length
        self.max_seq_length = max_seq_length
        self.hl_token = '<hl>'
        self.id = did
        
    def tokenize_base(self, tokenizer, shuffle_order=False):
        aligner = AnswerSpanAligner(self.context, tokenizer)
        self.aligner = aligner
        if shuffle_order:
            random.shuffle(self.answers)

        self.current_turn = 0
        self.questions = ["<extra_id_0>"] * len(self.answers)
        self.max_turn = len(self.answers)
    
    def add_question(self, question):
        self.questions[self.current_turn] = question
        self.current_turn += 1
        
    def is_end(self):
        return True if self.current_turn == self.max_turn else False
    
    def print_qas(self, original_questions=None):
        if not original_questions:
            original_questions = []

        for q, a, oq in zip(self.questions, self.answers, original_questions):
            if q == "<extra_id_0>":
                break

            print(f"Q: {q}  /  {oq}")
            print(f"A: {a['text']}")
        
    def prepare_input(self, tokenizer):
        if self.is_end():
            return
        
        a = self.answers[self.current_turn]
        passage = self.aligner.get_highlight_by_answer(tokenizer,
                                                       a['answer_start'],
                                                       a['text'],
                                                       self.max_passage_length,
                                                       self.hl_token
                                                     )
        passage = tokenizer.convert_tokens_to_ids(passage)
        available_length = self.max_seq_length - len(passage) - 3
        
        conv = []
        for q, a in zip(self.questions[:self.current_turn + 1], self.answers[:self.current_turn + 1]):
            a = a['text']

            conv.append(tokenizer.tokenize(q))
            conv.append(tokenizer.tokenize(a))

        context = []
        for idx, c in enumerate(conv):
            if context:
                context.append(tokenizer.eos_token)

            context.extend(c)

        context = tokenizer.convert_tokens_to_ids(context)

        if len(context) > available_length:
            gap = len(context) - available_length
            context = context[gap:]
        if tokenizer.cls_token_id is not None:
            cls = [tokenizer.cls_token_id]
        else:
            cls = []
        input_ = cls + passage + [tokenizer.eos_token_id] + context + [tokenizer.eos_token_id]
        
        return input_
    
    def to_dict(self):
        dic = {}
        for i, q in enumerate(self.questions):
            if q in ['<extra_id_0>']:
                continue
            
            dic[f"{self.id}_q#{i}"] = q
        return dic
