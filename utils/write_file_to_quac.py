
import json
import copy
import os
import re
import random
import numpy as np
import argparse
import glob

from copy import deepcopy

def get_did_dic(quac):
    did2idx = {}

    for i in range(len(quac['data'])):
        entry = quac['data'][i]
        did = entry['paragraphs'][0]['id']
        if did not in did2idx.keys():
            did2idx[did] = [i]
        else:
            did2idx[did] += [i]
    
    return did2idx


def find_str_start(ans_txt, context):
    if ans_txt == 'CANNOTANSWER':
        ans_start = len(context) - len('CANNOTANSWER')
    # if there is span matching with ans_txt
    else: 
        res = re.search(re.escape(ans_txt), context)
        ans_start = res.span()[0] if res else -1

    return ans_start


def write_jsonl_to_quac(result_dir):
    
    out_dials = []
    with open(os.path.join(result_dir, 'results.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            ins = json.loads(line)
            
            for idx_turn, qa in enumerate(ins['qas']):
                qa['id'] = ins['guid'] + f'_q#{idx_turn}'
                qa['orig_answer'] = qa['orig_answer'][0]
                
            out_para = {'context' : ins['context'],
                        'qas'     : ins['qas'],
                        'id'      : ins['guid'],
                        }
            out_dial = {'paragraphs'    : [out_para],
                        'section_title' : ins['section_title'],
                        'title'         : ins['title'],
                        'background'    : ins['background']}
            out_dials += [out_dial]

    out_quac = {'data' : out_dials}

    out_dir = os.path.join(result_dir, 'quac')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train.json'), 'w') as outfile:
        json.dump(out_quac, outfile)

    print(f'save quac file to: {out_dir}')


def write_file_to_quac(cqa_path, results_dir):
    
    quac = json.load(open(cqa_path, 'r'))

    did2idx = get_did_dic(quac)
    
    syn_dials = [] ; cnt = 0 ; err_cnt = 0

    with open(os.path.join(results_dir, 'results.jsonl'), 'r') as json_file:
        json_list = list(json_file)

    for txt_json in json_list:
        qa_reord = json.loads(txt_json)
        entry = {'title' : qa_reord['title']}
        did = qa_reord['guid'].split('_q#')[0]

        try:
            idx_entry = did2idx[did][0]
        except:
            print(f'{did} is not found in original dataset')
            continue

        para = quac['data'][idx_entry]['paragraphs'][0]
        qas = []
        for k, qa in enumerate(qa_reord['qas']):
            ans_text = qa['answer']['text']
            ans_start = qa['answer']['answer_start']
            
            if ans_text != para['context'][ans_start: ans_start + len(ans_text)]:
                print('span index is not matched. find it with string match')
                ans_start = find_str_start(ans_text, para['context'])
                if ans_start < 0:
                    err_cnt += 1
                    print('Fail to find answer start span :')
                    print(ans_text)
                    continue
                
            syn_ans = {'text' : ans_text,
                       'answer_start' : ans_start,
                      }
            
            syn_qa = {'id' : did + f"_q#{k}",
                    'question' : qa['question'],
                    'answers'  : [syn_ans] * 3,
                    'orig_answer' : syn_ans,
                    }
            qas += [deepcopy(syn_qa)]
            cnt += 1
        
        syn_para = {'id'      : did,
                    'context' : para['context'],
                    'qas'     : qas,
                }
        
        entry['paragraphs'] = [syn_para]
        syn_dials += [entry]
        
    quac_out = {'data' : syn_dials} 
    
    print(f'total {err_cnt} answers are removed due to index mismatch')
    print(f'total {cnt} questions are saved to {results_dir}/quac/train.json')
    out_dir = os.path.join(results_dir, 'quac')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train.json'), 'w') as f:
        json.dump(quac_out, f)
        
        
def write_jsonl_to_coqa(result_dir):
    out_dials = []
    with open(os.path.join(result_dir, 'results.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            temp = {
                'source': line['title'],
                'id': line['guid'],
                'filename': line['section_title'],
                'story': line['context'],
                'questions': [],
                'answers': [],
                'name': line['section_title']
            }
            for i, qa in enumerate(line['qas']):
                q = qa['question']
                a = qa['orig_answer'][0]
                temp['questions'].append({'input_text': q, 'turn_id': i + 1})
                temp['answers'].append(
                    {
                        'span_start': a['answer_start'],
                        'span_end': a['answer_start'] + len(a['text']) if a['text'] not in ['unknown', 'yes', 'no'] else -1,
                        'span_text': a['text'],
                        'input_text': a['text'],
                        'turn_id': i + 1
                    }
                )
            out_dials.append(temp)

    out_coqa = {'data' : out_dials}

    out_dir = os.path.join(result_dir, 'coqa')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train.json'), 'w') as outfile:
        json.dump(out_coqa, outfile)

    print(f'save coqa file to: {out_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert .jsonl files to QuAC format")
    parser.add_argument('--cqa_path', type=str, help='Path to dump results to')
    parser.add_argument('--results_dir', 
                        default='results/quac/unseen/toy',
                        type=str, help='Path to dump results to')
    
    args = parser.parse_args()
    result_path = os.path.join(args.results_dir, 'results.jsonl')
    print(result_path, 'is found.')

    # write_jsonl_to_quac(args.results_dir)
    write_file_to_quac(args.cqa_path, args.results_dir)
