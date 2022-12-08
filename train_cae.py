import os
import json
import torch
import argparse
from tqdm import tqdm
from argparse import Namespace

from transformers import BertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from cqg.cae_utils import AnswerExtractionDataset, compute_fuzzy_recall
from cqg import cae
from cqg.answer_extractor_model import AnswerSpanExtractor2DModel, postprocess_span2d_output
from util import set_seed, get_logger


logger = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
def evaluation(model, dev_loader, top_k=10, conversational=False):
    model.eval()
    all_outputs = []
    for bidx, batch in enumerate(dev_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            
            all_span_logits = outputs.span_logits.detach().cpu().numpy()
            all_span_masks = outputs.span_masks.detach().cpu().numpy()
        gt_answers = batch["answers"]
        for fidx in range(outputs[0].size(0)):
            p = postprocess_span2d_output(all_span_logits,
                                          all_span_masks,
                                          batch["passages"],
                                          batch["offsets"],
                                          fidx,
                                          30,
                                          top_k
                                         )
            answers = gt_answers[fidx]
            all_outputs.append([p, answers])
            
        if bidx % 100 == 0:
            print(f"Evaluation [{bidx}/{len(dev_loader)}]")

    hit = 0
    count = 0
    for i in range(len(all_outputs)):
        predicted_answers = [a["text"] for a in all_outputs[i][0]]
        gt_answers = []
        for q in all_outputs[i][1]:
            gt_answers.append([a["text"] for a in q["answers"]])
        h, c = compute_fuzzy_recall(predicted_answers, gt_answers, conversational=conversational)
        hit += h
        count += c
    recall = hit / count
    return recall, all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_history_length", type=int, default=32)
    parser.add_argument("--use_previous_qa", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.random_seed)
    args.n_gpu = torch.cuda.device_count()
    args.train_data_path = f"{args.data_path}/train.json"
    args.valid_data_path = f"{args.data_path}/valid.json"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    train_config = vars(args)
    json.dump(train_config, open(os.path.join(args.output_path, "train_config.json"), "w"), indent=2)
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = json.load(open(args.train_data_path))

    train_examples = []
    for line in tqdm(dataset["data"]):
        example = cae.make_instances(line,
                                     tokenizer,
                                     max_history_length=args.max_history_length,
                                     use_previous_qa=args.use_previous_qa,
                                     inference=False)
        train_examples.extend(example)

    train_data = AnswerExtractionDataset(train_examples, tokenizer, inference=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size,
                                               shuffle=True, collate_fn=train_data.collate_fn)

    dataset = json.load(open(args.valid_data_path))
    valid_examples = []
    for line in tqdm(dataset["data"]):
        example = cae.make_instances(line,
                                     tokenizer,
                                     max_history_length=args.max_history_length,
                                     use_previous_qa=args.use_previous_qa,
                                     inference=True)
        valid_examples.extend(example)

    valid_data = AnswerExtractionDataset(valid_examples, tokenizer, inference=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.eval_batch_size,
                                               shuffle=False, collate_fn=valid_data.collate_fn)

    model = AnswerSpanExtractor2DModel.from_pretrained(args.model_name_or_path)
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
    
    best_epoch = 0,
    best_score = -1
    for epoch in range(args.num_train_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            start_positions=batch["start_positions"],
                            end_positions=batch["end_positions"])

            loss = outputs.loss
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            if idx % 100 == 0:
                logger.info(f"[{epoch}/{args.num_train_epochs}] [{idx}/{len(train_loader)}] {loss.item()}")
        
        model_to_eval = model.module if hasattr(model, "module") else model
        recall, predictions = evaluation(model_to_eval, valid_loader, conversational=True)
        logger.info(f"Recall@10 at epoch {epoch}: {recall}")

        with open(f"{args.output_path}/output-{epoch}.json", "w") as f:
            json.dump(predictions, f, indent=2)
            f.write("\n")

        if recall >= best_score:
            logger.info(f"Update best model at epoch {epoch}")
            best_score = recall
            best_epoch = epoch
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
