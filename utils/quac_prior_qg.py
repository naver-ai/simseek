import torch
import logging
import json
import os

from typing import List, Optional
from tqdm import tqdm
from dataclasses import dataclass

from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ['<hl>']


@dataclass(frozen=False)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        guid: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    history: List[str]
    background: str
    qas: Optional[List]

    def update_q(self, question_text):
        self.qas = {'question' : question_text}

    def to_dict(self, turn_id, no_subt=False):
        init_qa = 1 if no_subt else 2
        subsection = self.history[1] if not no_subt else None
        dic = {
            'guid': self.guid,
            'title': self.history[0],
            'subsection': subsection,
            'history': self.history[init_qa:],
            'qas': self.qas,
            'turn_id': turn_id
        }
        return dic


class QuacQGProcessor:
    """Processor for the QuAC data set."""
    def __init__(self, num_labels=1):
        self.num_labels = num_labels

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        raise ValueError(
            "It can not be tested in current code"
        )
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "toy_train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "It can not be tested in current code"
        )
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "toy_dev.jsonl")), "dev")

    def get_mc_examples(self, data_dir, nm_file):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_mc_examples(self._read_jsonl(os.path.join(data_dir, nm_file)), "test")

    def get_qg_examples(self, data_dir, nm_file):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_qg_examples(json.load(open(os.path.join(data_dir, nm_file))), "test")

    def get_labels(self):
        """See base class."""
        return list(range(self.num_labels))

    def _read_jsonl(self, input_file):
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip('\n|\r')))
            return data

    def _create_qg_examples(self, input_data, type: str):
        """Creates examples for the training and dev sets."""
        examples = [
            InputExample(
                guid=dial['paragraphs'][0]['id'] + '_q#' + str(0),
                history=[dial['title'],
                         dial['section_title'],
                        ],
                background=dial['background'],
                qas=[],
            )
            for dial in tqdm(input_data['data'])  # we skip the line with the column names
        ]

        return examples
        
    def _create_mc_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = [
            InputExample(
                guid=line['meta']['quac_did'] + '_q#' + str(line['meta']['num_turn']),
                history=line['inputs']['history'],  # in the swag dataset, the
                background=line['inputs']['background'],
                qas=[],
            )
            for line in lines  # we skip the line with the column names
        ]

        return examples


class CQGDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.length = len(self.dataset)
        
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
        inputs, targets, guids = list(zip(*batch))
        input_ids = torch.tensor(self.pad_ids(inputs, self.tokenizer.pad_token_id))
        target_ids = torch.tensor(self.pad_ids(targets, self.tokenizer.pad_token_id))
        input_masks = input_ids.ne(self.tokenizer.pad_token_id)
        
        return input_ids, input_masks, target_ids, list(guids)


def load_processed_data(data_path):
    examples = []
    for line in open(data_path, 'r', encoding='utf-8'):
        examples.append(json.loads(line))
    return examples


def load_dataset(data_path, tokenizer, max_seq_length=512, max_passage_length=384, hl_token='<hl>', fewshot_t5=False,
                 use_answer=False, **kargs):
    data = json.load(open(data_path, 'r'))
    examples = []
    for d in tqdm(data['data']):
        example = make_instances(d, tokenizer, max_seq_length, max_passage_length, hl_token, fewshot_t5, use_answer)
        examples.extend(example)
    return examples


def make_instances(data,
                   tokenizer,
                   max_seq_length=512,
                   max_passage_length=384,
                   hl_token='<hl>',
                   fewshot_t5=True,
                   use_answer=False):
    instances = []
    title = data['title']
    subtitle = data['section_title']
    background = data['background']
    prior_context = f" {tokenizer.eos_token} ".join([title, subtitle, background])
    prior_context = tokenizer.tokenize(prior_context)
    prior_context = tokenizer.convert_tokens_to_ids(prior_context)
    
    for paragraph in data['paragraphs']:
        guid = None
        conv = []
        for tid, qas in enumerate(paragraph['qas']):
            a = qas['orig_answer']
            guid = qas['id']
            available_length = max_seq_length - len(prior_context) - 3

            a = a['text']
            q = qas['question']

            if fewshot_t5:
                conv.append(tokenizer.tokenize(q))
                conv.append(tokenizer.tokenize(a))
            else:
                conv.append(tokenizer.tokenize(a))
                conv.append(tokenizer.tokenize(q))

            context = []
            for idx, c in enumerate(conv[:-1]):
                if context:
                    context.append(tokenizer.eos_token)

                if fewshot_t5 and idx == len(conv) - 2:
                    c = ['<extra_id_0>', tokenizer.eos_token]  # target question
                context.extend(c)

            if fewshot_t5:
                if use_answer:
                    context.extend(conv[-1])  # target answer
                else:
                    context = context[:-1]

            context = tokenizer.convert_tokens_to_ids(context)

            if len(context) > available_length:
                gap = len(context) - available_length
                context = context[gap:]

            if fewshot_t5:
                target = tokenizer.encode(f"<extra_id_0> {q} <extra_id_1>")
            else:
                target = tokenizer.encode(q)

            if tokenizer.cls_token_id is not None:
                cls = [tokenizer.cls_token_id]
            else:
                cls = []

            if tokenizer.bos_token_id is None:
                # decoder_start_token_id for t5
                target = [tokenizer.pad_token_id] + target
            else:
                # decoder_start_token_id for bart
                target = [tokenizer.eos_token_id] + target[1:]

            input_ = cls + prior_context + [tokenizer.eos_token_id] + context + [tokenizer.eos_token_id]
            turn_guid = f"{guid}"
            instances.append([input_, target, turn_guid])
    return instances

def load_dataset_from_instances():
    examples = []


class CQGInstances(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        args,
        tokenizer,
    ):
        processor = QuacQGProcessor()
        
        self.turn_id        = 0
        self.max_seq_length = args.max_seq_length
        self.use_answer     = args.use_answer
        self.output_dir     = args.output_dir
        self.tokenizer      = tokenizer
        self.fewshot_t5     = True

        logger.info(f"Creating QG examples from dataset file at {args.data_dir}")
        if '.jsonl' in args.nm_file:
            self.examples = processor.get_mc_examples(args.data_dir, args.nm_file)
        else:
            self.examples = processor.get_qg_examples(args.data_dir, args.nm_file)
        logger.info("Training examples: %s", len(self.examples))

    def get_QG_dataset(self, split='test'):
        instances = []
        for ex in self.examples:
            title = ex.history[0]
            subtitle = ex.history[1]
            background = ex.background
            prior_context = f" {self.tokenizer.eos_token} ".join([title, subtitle, background])
            prior_context = self.tokenizer.tokenize(prior_context)
            prior_context = self.tokenizer.convert_tokens_to_ids(prior_context)
            prior_context = prior_context[:384]  # max background length

            available_length = self.max_seq_length - len(prior_context) - 3

            context = []
            conv = [self.tokenizer.tokenize(h) for h in ex.history[2:]] if len(ex.history) > 2 else []

            for idx, c in enumerate(conv):
                context.extend(c)

            if self.fewshot_t5:
                c = ['<extra_id_0>', self.tokenizer.eos_token]  # target question
                context.extend(c)

            context = self.tokenizer.convert_tokens_to_ids(context)

            if len(context) > available_length:
                gap = len(context) - available_length
                context = context[gap:]

            q = "" # test examples do not have target question
            if self.fewshot_t5:
                target = self.tokenizer.encode(f"<extra_id_0> {q} <extra_id_1>")
            else:
                target = self.tokenizer.encode(q)

            if self.tokenizer.cls_token_id is not None:
                cls = [self.tokenizer.cls_token_id]
            else:
                cls = []

            if self.tokenizer.bos_token_id is None:
                # decoder_start_token_id for t5
                target = [self.tokenizer.pad_token_id] + target
            else:
                # decoder_start_token_id for bart
                target = [self.tokenizer.eos_token_id] + target[1:]

            input_ = cls + prior_context + [self.tokenizer.eos_token_id] + context + [self.tokenizer.eos_token_id]
            turn_guid = f"{ex.guid}"
            instances.append([input_, target, turn_guid])

        return instances
    
        
    def save_all_examples(self):
        for example in self.examples:
            with open(os.path.join(self.output_dir, 'results.jsonl'), "a", encoding="utf-8") as f:
                f.write(json.dumps(example.to_dict(self.turn_id)) + '\n')

        print("all examples are sucessfully saved to :")
        print(self.output_dir)

    def _pop_and_save_examples(self, guid_to_ended):
        done = 0
        for idx_e in range(len(self.examples) - 1, -1, -1):
            example = self.examples[idx_e]
            if not guid_to_ended[example.guid]:
                continue

            with open(os.path.join(self.output_dir, 'results.jsonl'), "a", encoding="utf-8") as f:
                f.write(json.dumps(example.to_dict(self.turn_id, self.no_subt)) + '\n')
            self.examples.pop(idx_e)
            done += 1
            
        print(f"[{done} examples done!!]")

    def _update_examples_and_features(
            self, 
            guid_to_target_cqa, 
            ):

        # update examples
        for idx, example in enumerate(self.examples):
            target_cqa = guid_to_target_cqa[example.guid]
                
            example.history += [target_cqa['question'], target_cqa['answer']['text']]
            guid_qsplit = example.guid.split('_q#')
            example.guid = guid_qsplit[0] + '_q#' + str(self.turn_id + 1) #+ '_d#' + guid_qdsplit[1]        
            example.qas += [target_cqa]

        len_exmpls = len(self.examples)
        print(f'{len_exmpls} examples are left')
        self.turn_id += 1
