# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on QuAC (RoBERTa)."""


import os
import re
import argparse
import glob
import copy
import logging
import random
import timeit
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
)
from transformers import (
    T5Tokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)

from run_cqa import (
    evaluate,                            
)
from cqa.quac_processors import quac_convert_examples_to_features, QuacProcessor

from utils.quac_prior_qg import (
    load_dataset, 
    SPECIAL_TOKENS, 
    CQGDataset,
    CQGInstances,
)
from utils.write_file_to_quac import write_file_to_quac
from utils.post_filtering import post_processing_question
from utils.data_utils import set_seed


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_random_labels(instances):
    guid_to_label, guid_to_ended = {}, {}
    
    for ex in instances.examples:
        num_endings = len(ex.endings)
        guid_to_label[ex.guid] = random.choice(list(range(num_endings)))
        ## TODO : random order + sampling
        guid_to_ended[ex.guid] = False
    
    return guid_to_label, guid_to_ended


def _pop_element_with_indices(lst, indices_pop):
    for idx_pop in sorted(indices_pop, reverse=True):
        lst.pop(idx_pop)

def concat_history(tokenizer, qas, turn_t, max_history=1):
    question_text = ""
    sep_token = tokenizer.sep_token
    tokenizer_name = tokenizer.__class__.__name__
    if tokenizer_name in ["RobertaTokenizer", "RobertaTokenizerFast"]:
        sep_token += tokenizer.sep_token
    
    idx_start_turn = turn_t - max_history
    if max_history > 0 and len(qas) > 1 and idx_start_turn >= 0:
        history = qas[2 * idx_start_turn: ]
        question_text += sep_token + sep_token.join(history)

    return question_text

def _append_next_questions(tokenizer, cqa_examples, guid_to_cand_qs):
    
    # append obtained questions to cqa example by using the dictionary
    idxs_pop = []
    new_exs = []
    for idx, example in enumerate(cqa_examples):
        if '_q#' in example.qas_id:
            did, turn_num = example.qas_id.split('_q#')
            example.qas_id = did + '_q#' + str(int(turn_num) + 1)
            example.question_text = concat_history(tokenizer, example.history, int(turn_num) + 1)
        else: # have no question yet
            example.qas_id += f'_q#{0}'
        
        if example.qas_id not in guid_to_cand_qs.keys():
            idxs_pop += [idx]
            logger.warning(
            "WARNING - some examples' ids are not incldued in guids"
            )
            continue

        next_q = guid_to_cand_qs[example.qas_id]['question'][0]
        num_cands = len(guid_to_cand_qs[example.qas_id]['question'])
        if num_cands > 1: # args.cqg_num_returen_seq > 1
            for idx_cand in range(1, num_cands):
                new_ex = copy.deepcopy(example)
                new_ex.question_text = guid_to_cand_qs[example.qas_id]['question'][idx_cand] + new_ex.question_text
                new_ex.qas_id += f'_c#{idx_cand}'
                new_exs += [new_ex]
            example.qas_id += '_c#0'

        example.question_text = next_q + example.question_text
    
    _pop_element_with_indices(cqa_examples, idxs_pop)
    cqa_examples += new_exs


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--cqa_model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--cqa_model_name_or_path",
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
    )

    # Other parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--cqa_max_seq_length",
        default=512,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--max_query_length",
        default=128,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_not_validate", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=12, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=100, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--max_history",
        default=1,
        type=int,
        help="The maximum number of conversation history which will be concated to the current question "
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed."
        "A number of warnings are expected for a normal QuAC evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--valid_steps", type=int, default=0, help="Validate the model every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--threads", type=int, default=10, help="multiple threads for converting example to features")

    ## Reorder model args
    parser.add_argument('--cqg_model_name_or_path', type=str, required=True)
    parser.add_argument('--cqg_tokenizer_name', type=str, default=None)

    # experiments
    parser.add_argument('--find_until', type=str, default='max_turn',
                        help="Choose when the framework stop"
                            +"'all_choices' : exhaustive search for all choices each instance has"
                            +"'max_turn': terminate conversations when it reaches the maximum turn (--max_turn)")
    parser.add_argument('--filter', type=str, default=None,
                        help="Choose how to select thefinal QA pair"
                            +"None : no filter" 
                            +"'random' : randomly sample" 
                            )
    parser.add_argument('--max_turn', type=int, default=-1)
    parser.add_argument('--use_answer', action='store_true', default=False)
    parser.add_argument('--no_subt', action='store_true', default=False)
    parser.add_argument('--cqg_batch_size', type=int, default=2)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--top_p', type=float, default=0.98)
    parser.add_argument('--cqg_num_return_seq', type=int, default=1)

    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.overwrite_output_dir:
        output_path = os.path.join(args.output_dir, 'results.jsonl')
        if glob.glob(output_path):
            print(f'Following output file is found. overwrite on: {output_path}')
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('')
    
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args.seed)

    # Load pretrained model for cqg model
    qg_tokenizer = T5Tokenizer.from_pretrained(args.cqg_model_name_or_path)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(args.cqg_model_name_or_path)
    qg_model.to(device)

    # Load pretrained model for CQA
    args.model_type = args.cqa_model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.cqa_model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.cqa_model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.cqa_model_name_or_path,
        from_tf=bool(".ckpt" in args.cqa_model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Load instances for cqa
    cqg_instances = (
        CQGInstances(
            args=args,
            tokenizer=qg_tokenizer,
        )
    )
    
    # Load examples from cqa datasets
    processor = QuacProcessor(
                tokenizer=tokenizer,
                max_history=args.max_history,
                )
    cqa_examples = processor.get_doc_examples(args.data_dir, filename=args.nm_file)
    
    print(f'iterations started : maximum turn = {args.max_turn}')
    turn_id = 0
    while turn_id < args.max_turn:
        # 1. Question Generation
        qg_data = cqg_instances.get_QG_dataset()

        qg_dataset = CQGDataset(qg_data, qg_tokenizer)
        qg_sampler = SequentialSampler(qg_dataset)
        qg_loader = DataLoader(qg_dataset, batch_size=args.cqg_batch_size,
                                sampler=qg_sampler, collate_fn=qg_dataset.collate_fn,
                                num_workers=args.threads)

        guid_to_cand_qs = {}
        for batch in tqdm(qg_loader):
            input_ids, input_masks, target_ids, guids = (b if isinstance(b, list) else b.to(device) for b in batch)

            outputs = qg_model.generate(input_ids=input_ids,
                                        attention_mask=input_masks,
                                        num_beams=args.beam_size,
                                        num_return_sequences=args.cqg_num_return_seq,
                                        do_sample=True,
                                        top_p=args.top_p,
                                        temperature=args.temperature,
                                        max_length=args.max_length,
                                        length_penalty=1
                                        )
            
            generated = []
            for output in qg_tokenizer.batch_decode(outputs, skip_special_tokens=True):
                output = post_processing_question(output)
                generated.append(output)
            
            if args.cqg_num_return_seq > 1:
                questions = []
                for idx_b in range(args.cqg_batch_size):
                   questions += [generated[args.cqg_num_return_seq * idx_b: 
                                          args.cqg_num_return_seq * (idx_b + 1)]
                                ]
            else:
                questions = [[q] for q in generated]

            for guid, question in zip(guids, questions):
                guid_to_cand_qs[guid] = {'question' : question}

        # 2. Append selected questions and load cqa examples
        _append_next_questions(tokenizer, 
                               cqa_examples,
                               guid_to_cand_qs,
                               )

        features, dataset = quac_convert_examples_to_features(
            examples=cqa_examples,
            tokenizer=tokenizer,
            max_seq_length=args.cqa_max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=args.threads,
        )

        # 3. Predict answers for questions and append them
        preds, nbest, ans_starts = evaluate(args, model, tokenizer, output_dir, 
                                        dataset, cqa_examples, features, inference=True)

        guid_to_target_qa = {}
        if args.cqg_num_return_seq > 1:
            guid_to_cands = {}
            qid_to_pred = {qid : pred for qid, pred in preds.items()}
            qid_to_start = {qid : start for qid, start in ans_starts.items()}
            idxs_pop = []
            for idx_ex, ex in enumerate(cqa_examples):
                guid, idx_cand = ex.qas_id.split(f"_c#")
                cand_meta = {'idx_ex'   : int(idx_ex),
                             'idx_cand' : int(idx_cand)}

                if guid not in guid_to_cands.keys():
                    guid_to_cands[guid] = [cand_meta]
                else:
                    guid_to_cands[guid] += [cand_meta]
            
            for guid, lst_cands in guid_to_cands.items():
                weights = [1] * len(lst_cands)
                idx_slt = random.choices(list(range(len(lst_cands))), weights)[0]
                slt_cand = lst_cands.pop(idx_slt)

                slt_ex = cqa_examples[slt_cand['idx_ex']]
                cur_q_text = guid_to_cand_qs[guid]['question'][slt_cand['idx_cand']]
                pred = qid_to_pred[slt_ex.qas_id]
                ans_start = qid_to_start[slt_ex.qas_id]
                guid_to_target_qa[guid] = {
                                            'question' : cur_q_text,
                                            'answer'   : {"text" : pred,
                                                          "answer_start" : ans_start},
                                            }
                slt_ex.qas_id = guid
                idxs_pop += [cand['idx_ex'] for cand in lst_cands]
            _pop_element_with_indices(cqa_examples, idxs_pop)

        else:
            for (id_pred, pred), (_, ans_start) in zip(preds.items(), ans_starts.items()):
                assert id_pred == _
                guid_to_target_qa[id_pred] = {
                                             'question' : guid_to_cand_qs[id_pred]['question'][0],
                                             'answer'   : {"text" : pred,
                                                           "answer_start" : ans_start},
                                             }

        # 4. update examples and terminate the current turn
        for ex in cqa_examples:
            ex.history += [guid_to_target_qa[ex.qas_id]['question'],
                           guid_to_target_qa[ex.qas_id]['answer']['text'],
                          ]
                          
        cqg_instances._update_examples_and_features(guid_to_target_qa)

        turn_id += 1
        print(f'turn ended : {turn_id} / {args.max_turn}')

    cqg_instances.save_all_examples()
    cqa_path = os.path.join(args.data_dir, args.nm_file)
    write_file_to_quac(cqa_path, args.output_dir)


if __name__ == "__main__":
    random.seed(0)
    main()
