import os
import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler

from transformers import T5Tokenizer, BartTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.metrics import BLEU
from cqg.cqg_utils import HL_TOKEN, SPECIAL_TOKENS, CQGDataset
from cqg import answer_cqg, prior_cqg
from util import set_seed, get_logger
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(__name__)
        
        
def evaluation(model,
               dev_loader,
               tokenizer,
               num_beams=4,
               max_length=32,
               length_penalty=1):
    model.eval()
    metric = BLEU()
    predictions = [["guid", "pred", "truth"]]
    for batch in dev_loader:
        input_ids, attention_mask, target_ids, guids = (b if isinstance(b, list) else b.to(device) for b in batch)

        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 num_beams=num_beams,
                                 max_length=max_length,
                                 length_penalty=length_penalty
                                 )
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        truths = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        for pred, truth, guid in zip(preds, truths, guids):
            metric.update((pred, truth))
            predictions.append([guid, pred, truth])
    result = metric.compute()
    return result, predictions

    
def train_cqg(args,
              model,
              tokenizer,
              train_loader,
              optimizer,
              scheduler,
              dev_loader,
              test_loader,
              do_early_stop=False,
              patience=10,
              metric_key="bleu-4",
              logging_step=100):

    scaler = GradScaler()
    best_epoch = 0
    best_score = -1
    global_step = 0
    eval_result = {}
    predictions = []
    _patience_count = 0
    
    while len(train_loader) < logging_step:
        if logging_step <= 1:
            logging_step = 1
            break

        logging_step = int(logging_step / 10)
    
    for epoch in range(args.num_train_epochs):
        model.train()
        
        if do_early_stop and _patience_count > patience:
            logger.info("Early stopping is activated...! Exit training")
            break

        for idx, batch in enumerate(train_loader):
            input_ids, attention_mask, target_ids, guids = (b if isinstance(b, list) else b.to(device) for b in batch)
            y_ids = target_ids[:, :-1].contiguous()
            lm_labels = target_ids[:, 1:].clone()
            lm_labels[lm_labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            if args.fp16:
                with autocast():
                    outputs = model(input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=y_ids,
                                    labels=lm_labels)
            else:
                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids=y_ids,
                                labels=lm_labels)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()
            
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            global_step += 1

            if idx % logging_step == 0:
                logger.info(f"[{epoch}/{args.num_train_epochs}][{idx}/{len(train_loader)}] {loss.item()}")

        model_to_eval = model.module if hasattr(model, "module") else model
        eval_result, predictions = evaluation(model_to_eval,
                                              dev_loader,
                                              tokenizer,
                                              num_beams=4,
                                              max_length=32,
                                              length_penalty=1)
        logger.info(f"{epoch} evaluation result: {eval_result}")

        if eval_result[metric_key] >= best_score:
            best_score = eval_result[metric_key]
            best_epoch = epoch
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_path)
            with open(f"{args.output_path}/predictions_{epoch}.txt", "w", encoding="utf-8") as f:
                for prediction in predictions:
                    f.write("\t".join(prediction) + "\n")

            _patience_count = 0
            logger.info(f"Update best score {best_score}, flush patience... {_patience_count}")
        else:
            _patience_count += 1
            logger.info(f"Stacking patience... {_patience_count}")

    logger.info(f"Start test based on {best_epoch}...")
    model_to_eval = model.module if hasattr(model, "module") else model
    model_to_eval = model_to_eval.from_pretrained(args.output_path)
    model_to_eval = model_to_eval.to(device)
    eval_result, predictions = evaluation(model_to_eval,
                                          test_loader,
                                          tokenizer,
                                          num_beams=4,
                                          max_length=32,
                                          length_penalty=1)
    logger.info(f"{best_epoch} test evaluation result: {eval_result}")
    model = model_to_eval
    return model, predictions, eval_result


def main(args):
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.random_seed)
    if "bart" in args.model_name_or_path.lower():
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    
    if args.task.startswith("prior"):
        target_task = prior_cqg
    else:
        target_task = answer_cqg

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
        
    tokenizer.add_tokens(SPECIAL_TOKENS)
    tokenizer.save_pretrained(args.output_path)

    train_config = vars(args)
    json.dump(train_config, open(os.path.join(args.output_path, "train_config.json"), "w"), indent=2)

    train_examples = target_task.load_dataset(f"{args.data_path}/train.json",
                                              tokenizer,
                                              args.max_seq_length,
                                              args.max_passage_length,
                                              HL_TOKEN,
                                              args.fewshot_t5)
    train_data = CQGDataset(train_examples, tokenizer)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size,
                              sampler=train_sampler, collate_fn=train_data.collate_fn)
    
    dev_examples = target_task.load_dataset(f"{args.data_path}/valid.json",
                                            tokenizer,
                                            args.max_seq_length,
                                            args.max_passage_length,
                                            HL_TOKEN,
                                            args.fewshot_t5)
    dev_data = CQGDataset(dev_examples, tokenizer)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(dev_data, batch_size=args.eval_batch_size,
                             sampler=dev_sampler, collate_fn=dev_data.collate_fn)

    test_examples = target_task.load_dataset(f"{args.data_path}/dev.json",
                                             tokenizer,
                                             args.max_seq_length,
                                             args.max_passage_length,
                                             HL_TOKEN,
                                             args.fewshot_t5)
    test_data = CQGDataset(test_examples, tokenizer)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size,
                             sampler=test_sampler, collate_fn=test_data.collate_fn)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    t_total = len(train_loader) * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(t_total * args.warmup_ratio), num_training_steps=t_total
            )
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model, predictions, eval_result = train_cqg(args,
                                                model,
                                                tokenizer,
                                                train_loader,
                                                optimizer,
                                                scheduler,
                                                dev_loader,
                                                test_loader,
                                                do_early_stop=False,
                                                patience=args.patience)

    with open(f"{args.output_path}/predictions.txt", "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write("\t".join(prediction) + "\n")

    for k, v in eval_result.items():
        logger.info(f"{k}: {v}, avg: {np.mean(v)}")

    with open(f"{args.output_path}/eval_result.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None, required=True)
    parser.add_argument("--task", type=str, default="prior_cqg")
    parser.add_argument("--model_name_or_path", type=str, default="t5-large")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_passage_length", type=int, default=384)
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="training with mixed precision")
    parser.add_argument("--fewshot_t5", action="store_true", default=False)
    parser.add_argument("--do_early_stop", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    main(args)
