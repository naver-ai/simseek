import os
import random
import torch
import json
from tqdm import tqdm

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizerFast
from cqa.answer_extractor import AnswerSpanExtractor2DModel, postprocess_span2d_output
from utils.information_symmetric_cqg import ISSequentialCQGInstance
from utils.write_file_to_quac import write_jsonl_to_quac, write_jsonl_to_coqa
from utils.post_filtering import post_processing_question
from utils.data_utils import set_seed

import argparse


def load_dataset(data_path, cae_tokenizer, cqg_tokenizer, task="quac"):
    data = json.load(open(data_path, 'r'))
    instances = []
    for dial in tqdm(data['data'], desc=f"read {task} examples and load to instances"):
        instance = ISSequentialCQGInstance(
                                        title          = dial.get('title') or dial.get("source", ""),
                                        section_title  = dial.get('section_title') or dial.get("filename", ""),
                                        background     = dial.get('background'),
                                        context        = dial['paragraphs'][0]['context'] if "paragraphs" in dial else dial["story"], 
                                        history        = [],
                                        qas            = [],
                                        guid           = dial['paragraphs'][0]['id'] if "paragraphs" in dial else dial["id"],
                                        task           = task
                                        )
        instance.set_aligner(cae_tokenizer, cqg_tokenizer)
        instances += [instance]

    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="quac",
        type=str,
        help="Name of target task. Currently support [quac|coqa]",
    )
    parser.add_argument(
        "--cqg_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--cae_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
    )
    parser.add_argument(
        "--nm_file",
        default="train.json",
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False)

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--shuffle_answers', action='store_true', default=False)
    parser.add_argument('--top_k_answer', type=int, default=5)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=0.98)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--max_turn', type=int, default=-1)
    parser.add_argument('--add_cannot_answer', action='store_true', default=True)
    parser.add_argument('--max_cannot_answer', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    assert args.task in ["quac"]
    
    set_seed(args.seed)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # prepare models (CQG & CAE)
    cqg_model = AutoModelForSeq2SeqLM.from_pretrained(args.cqg_model_name_or_path)
    cqg_model.to(device)
    cqg_tokenizer = T5Tokenizer.from_pretrained(args.cqg_model_name_or_path)

    cae_model = AnswerSpanExtractor2DModel.from_pretrained(args.cae_model_name_or_path)
    cae_model.to(device)
    cae_tokenizer = BertTokenizerFast.from_pretrained(args.cae_model_name_or_path)

    data_path = os.path.join(args.data_dir, args.nm_file)
    instances = load_dataset(data_path, cae_tokenizer, cqg_tokenizer, args.task)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.overwrite_output_dir:
        with open(os.path.join(args.output_dir, 'results.jsonl'), 'w') as f:
            print(f'overwrite on the following path: \n{args.output_dir}')
        
    ## args
    top_k_answer = 5
    max_turn = args.max_turn
    
    turn_id = 0
    while turn_id < max_turn:
        # 1. Extract prior answer from given context C and history H_{t-1}
        for instance in tqdm(instances, desc=f'Conversational Answer Extractor'):
            cae_instance = instance.make_cae_input(cae_tokenizer)
            input_id = torch.tensor([cae_instance.input_id])
            attention_masks = input_id.ne(cae_tokenizer.pad_token_id).float()
            with torch.no_grad():
                outputs = cae_model(input_id.to(device), attention_masks.to(device))

            all_span_logits = outputs.span_logits.detach().cpu().numpy()
            all_span_masks = outputs.span_masks.detach().cpu().numpy()

            for fidx in range(outputs[0].size(0)):
                p = postprocess_span2d_output(all_span_logits,
                                              all_span_masks,
                                              [cae_instance.passage],
                                              [cae_instance.offset],
                                              fidx,
                                              30,
                                              top_k_answer
                                              )

            instance.set_answer_priors(p,
                                        add_cannot_answer=args.add_cannot_answer,
                                        max_cannot_answer=args.max_cannot_answer)
        
        # 2. Generate answer-aware question from given C and prior answer a_t
        for instance in tqdm(instances, desc=f'Question Generation'):
            input_id, ans_txt, ans_start = instance.make_cqg_input(cqg_tokenizer)

            outputs = cqg_model.generate(input_id.to(device),
                                         do_sample=True,
                                         num_beams=args.num_beams,
                                         top_p=args.top_p,
                                         temperature=args.temperature,
                                         max_length=args.max_length,
                                         length_penalty=1)
            q = cqg_tokenizer.decode(outputs[0], skip_special_tokens=True)
            q = post_processing_question(q)
            instance.add_context(q)
            instance.add_context(ans_txt)
            
            answer = {'text'         : ans_txt,
                      'answer_start' : ans_start}
            qa     = {'question'     : q,
                      'answers'      : [answer],
                      'orig_answer'  : [answer]}
            instance.add_qa(qa)
        
        print(f'turn ended : {turn_id + 1} / {max_turn}')

        turn_id += 1
    
    ## save all instances
    out_path = os.path.join(args.output_dir, 'results.jsonl')    
    print(f'save results to the following path : \n {out_path}')
    for instance in instances:
        instance.save_to_jsonl(out_path)
    
    write_jsonl_to_quac(args.output_dir)

    
if __name__ == '__main__':
    main()
