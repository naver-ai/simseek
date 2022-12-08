from .span import AnswerSpanAligner
import json
from tqdm import tqdm


def load_processed_data(data_path):
    examples = []
    for line in open(data_path, "r", encoding="utf-8"):
        examples.append(json.loads(line))
    return examples


def load_dataset(data_path, tokenizer, max_seq_length=512, max_passage_length=384, hl_token="<hl>", fewshot_t5=False, **kargs):
    data = json.load(open(data_path, "r"))
    examples = []
    for d in tqdm(data["data"]):
        example = make_instances(d, tokenizer, max_seq_length, max_passage_length, hl_token, fewshot_t5)
        examples.extend(example)
    return examples


def make_instances(data,
                   tokenizer,
                   max_seq_length=512,
                   max_passage_length=384,
                   hl_token="<hl>",
                   fewshot_t5=False):
    instances = []
    
    title = data["title"]
    subtitle = data["section_title"]
    prior_context = f" {tokenizer.eos_token} ".join([title, subtitle])
    prior_context = tokenizer.tokenize(prior_context)
    prior_context = tokenizer.convert_tokens_to_ids(prior_context)
    
    for paragraph in data["paragraphs"]:
        aligner = AnswerSpanAligner(paragraph["context"], tokenizer)
        guid = None
        conv = []
        for tid, qas in enumerate(paragraph["qas"]):
            a = qas["orig_answer"]
            guid = qas["id"]
            passage = aligner.get_highlight_by_answer(tokenizer,
                                                      a["answer_start"],
                                                      a["text"],
                                                      max_passage_length,
                                                      hl_token
                                                     )
            passage = tokenizer.convert_tokens_to_ids(passage)
            passage = prior_context + [tokenizer.eos_token_id] + passage
            available_length = max_seq_length - len(passage) - 3

            a = a["text"]
            q = qas["question"]

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
                    c = ["<extra_id_0>", tokenizer.eos_token]  # target question
                context.extend(c)

            if fewshot_t5:
                context.extend(conv[-1])  # target answer

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

            input_ = cls + passage + [tokenizer.eos_token_id] + context + [tokenizer.eos_token_id]
            turn_guid = f"{guid}"
            instances.append([input_, target, turn_guid])
    return instances
