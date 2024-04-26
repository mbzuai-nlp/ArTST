import re
import sys
import argparse
import unicodedata
import os 
import pyarabic.araby as araby

map_numbers = {'0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'}
map_numbers = dict((v, k) for k, v in map_numbers.items())
punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])
punctuations = punctuations + '÷#ݣ+=|$×⁄<>`åûݘ ڢ̇ پ'


def convert_numerals_to_digit(word):
    sentence=[]
    for w in word:
        sentence.append(map_numbers.get(w, w))
    word = ''.join(sentence)
    return word

def remove_diacritics(word):
    return araby.strip_diacritics(word)

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()

def normalize_text(sentence):
    sentence = convert_numerals_to_digit(sentence)
    sentence = remove_diacritics(sentence)
    sentence = remove_punctuation(sentence)
    return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",'--input_file', required=True, help='Input file with text')
    parser.add_argument("-o",'--output_file', required=True, help='Output  file for normalized text')
    args = parser.parse_args()
    
    out = open(args.out_file, 'w')
    with open(args.input_file, 'r') as f:
        for line in f:
            sentence = line.strip()
            normalized_sentence = normalize_text(sentence)
            print(f"{normalized_sentence}", file=out)
    out.close()