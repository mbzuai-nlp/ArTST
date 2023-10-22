import re
import sys

import unicodedata
import argparse
from evaluate import load

from loguru import logger


def remove_punctuation(word):
    return word.translate(
        str.maketrans('', '', re.sub('[@% ]','', punctuations))
    ).lower()
    
def read_texts(path):
    with open(path, 'r') as file:
        lines_list = file.readlines() 
    return lines_list, [line.replace(' ','') for line in lines_list]

def read_wer_texts(path):
    with open(path, 'r') as file:
        lines_list = file.readlines() 
    return lines_list

def compute_cer(ref, pred, metric):
    return load('cer').compute(predictions=pred, references=ref)

def compute_wer(ref, pred):
    return load('wer').compute(predictions=pred, references=ref)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t','--true_path', type=str)
    parser.add_argument('-p','--pred_path', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-m', '--metric', type=str, default='cer')

    args = parser.parse_args()
    assert args.metric in ['wer', 'cer']

    punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])
    punctuations = punctuations + '#÷#ݣ+=|$×⁄<>`åûݘ ڢ̇ پ\n'

    if args.metric == 'cer':
        space_true, no_space_true = read_texts(args.true_path)
        space_pred, no_space_pred = read_texts(args.pred_path)

        space_cer = compute_cer(space_true,space_pred )
        no_space_cer = compute_cer(no_space_true, no_space_pred)

        logger.info(f"spaced CER for {args.dataset} : {space_cer} \n CER without space : {no_space_cer}")

    else:
        true = read_wer_texts(args.true_path)
        pred = read_wer_texts(args.pred_path)
        
        wer = compute_wer(true, pred)
        logger.info(f"WER for {args.dataset} : {wer}")
