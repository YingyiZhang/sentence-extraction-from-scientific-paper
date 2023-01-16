import argparse
import os
import transformers
from nltk.tokenize import sent_tokenize

TOKENIZER_NAMES = ['roberta-base', 'facebook/bart-large', 'bert-base-uncased', 'word', 'allenai/scibert_scivocab_uncased']

def init_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-input_dir',
                        default='dataset/semeval/', help='Path to jsonl files.')
    parser.add_argument('--output_dir', '-output_dir',
                        default='dataset/semeval/scibert/')
    parser.add_argument('--input_file', '-input_file',
                        default='train0',
                        help='Path to jsonl files.')
    parser.add_argument('--input_text', '-input_text',
                        default=None,
                        help='use input text')
    parser.add_argument('--langModel', '-langModel', default='allenai/scibert_scivocab_uncased', choices=TOKENIZER_NAMES, help='.')
    parser.add_argument('--lowercase', '-lowercase', default=True, choices=TOKENIZER_NAMES, help='.')
    parser.add_argument('--features', '-features', default=False)
    parser.add_argument('--need_sen_tokenize','-need_sen_tokenize', default=False)
    parser.add_argument('--original_file', '-original_file', default=False)
    opt = parser.parse_args()

    return opt


def read_ori_text(text, need_sen_tokenize=False):
    if need_sen_tokenize:
        _line = sent_tokenize(line)
        return _line
    else:
        return text

def read_ori_file(file, need_sen_tokenize=False):
    with open(file, 'r', encoding='utf-8', errors='ignore') as fr:
        lines = fr.readlines()
        if need_sen_tokenize:
            new_lines = []
            for line in lines:
                _line = sent_tokenize(line)
                new_lines.extend(_line)
            return new_lines
        else:
            return lines

def read_ori_dataset(file):
    with open(file, 'r', encoding='utf-8', errors='ignore') as fr:
        lines=fr.readlines()
        ids = []
        sens = []
        tags = []
        for line in lines:
            line = line
            line_sps = line.strip().split("\t")

            id = line_sps[0]
            sen = line_sps[1].split()
            tag = line_sps[2]
            ids.append(id)
            sens.append(sen)
            tags.append(tag)

    return ids, sens, tags

def get_pretrained_language_model(langModel,lowercase):
    if langModel.count("bert")>0 and langModel.count('roberta')==0:
        pretrained_tokenizer = transformers.BertTokenizer.from_pretrained(langModel, lowercase=lowercase)
        tokenizer_fn = pretrained_tokenizer.tokenize
    if langModel.count('roberta')>0:
        pretrained_tokenizer = transformers.RobertaTokenizer.from_pretrained(langModel, lowercase=lowercase)
        tokenizer_fn = pretrained_tokenizer.tokenize
    if langModel.count('bart')>0:
        pretrained_tokenizer = transformers.BartTokenizer.from_pretrained(langModel, lowercase=lowercase)
        tokenizer_fn = pretrained_tokenizer.tokenize
    return tokenizer_fn

def process_word(word, langModel, tokenizer, lowercase=True):
    #tokenizer_fn = get_pretrained_language_model(tokenizer, lowercase)#take this outside, downloadOne
    if langModel.count("roberta")>0:
        subwords = tokenizer(word, add_prefix_space=True)
    else:
        subwords = tokenizer(word)
    return subwords

def process_sen(sen, langModel, tokenizer, lowercase=True, poss=False):
    tokenizer_sentence = []
    if poss==False:
        for word in sen:
            subwords = process_word(word, langModel, tokenizer, lowercase)
            tokenizer_sentence.extend(subwords)
        return tokenizer_sentence
    else:
        for word in sen:
            subwords = process_word(word, langModel, tokenizer, lowercase)
            tokenizer_sentence.extend(subwords)
        return tokenizer_sentence

def main_fuction(input_dir=False, output_dir=False, input_file=False, features=False, input_text=False, need_sen_tokenize=False, original_file=False):
    input_dir = input_dir
    output_dir = output_dir
    input_file = input_file
    input_text = input_text
    need_sen_tokenize = need_sen_tokenize
    original_file = original_file

    tokenizer_fn = get_pretrained_language_model(opt.langModel, lowercase=opt.lowercase)


    # for item in items:
    if input_text:
        if need_sen_tokenize:
            lines = read_ori_text(input_text, need_sen_tokenize)
        else:
            lines = read_ori_text(input_text)

        if original_file:
            sens = lines
        else:
            ids, sens, tags = read_ori_dataset(lines)
    else:
        if need_sen_tokenize:
            lines = read_ori_text(os.path.join(input_dir, input_file), need_sen_tokenize)
        else:
            lines = read_ori_text(os.path.join(input_dir, input_file))

        if original_file:
            sens = lines
        else:
            ids, sens, tags = read_ori_dataset(lines)

    output_lines = []
    with open(os.path.join(output_dir, input_file), 'w') as fw:
        for i, line in enumerate(sens):
            if original_file:
                tokenizer_sen = process_sen(line, opt.langModel, tokenizer_fn,
                                            lowercase=opt.lowercase)
                output_lines.append(' '.join(tokenizer_sen))
                fw.write(' '.join(tokenizer_sen) + "\n")

            else:
                tokenizer_sen = process_sen(line, opt.langModel, tokenizer_fn,
                                                lowercase=opt.lowercase)
                output_lines.append(
                        ids[i] + "\t" + ' '.join(tokenizer_sen) + "\t" + tags[i] + "\n")
                fw.write(ids[i] + "\t" + ' '.join(tokenizer_sen) + "\t" + tags[i] + "\n")

    return output_lines

if __name__ == "__main__":
    opt = init_opt()
    input_dir = opt.input_dir
    output_dir = opt.output_dir
    input_file = opt.input_file
    input_text = opt.input_text
    need_sen_tokenize = opt.need_sen_tokenize
    original_file = opt.original_file

    tokenizer_fn = get_pretrained_language_model(opt.langModel, lowercase=opt.lowercase)

    #for item in items:
    if input_text:
        if need_sen_tokenize:
            lines = read_ori_text(input_text, need_sen_tokenize)
        else:
            lines = read_ori_text(input_text)

        if original_file:
            sens = lines
        else:
            ids, sens, tags = read_ori_dataset(lines)
    else:
        if need_sen_tokenize:
            lines = read_ori_text(os.path.join(input_dir, input_file), need_sen_tokenize)
        else:
            lines = read_ori_text(os.path.join(input_dir, input_file))

        if original_file:
            sens = lines
        else:
            ids, sens, tags = read_ori_dataset(lines)


    output_lines = []
    with open(os.path.join(output_dir, input_file), 'w') as fw:
        for i, line in enumerate(sens):
            if original_file:
                tokenizer_sen = process_sen(line, opt.langModel, tokenizer_fn,
                                            lowercase=opt.lowercase)
                output_lines.append(' '.join(tokenizer_sen))
                fw.write( ' '.join(tokenizer_sen)+ "\n")

            else:
                tokenizer_sen = process_sen(line, opt.langModel, tokenizer_fn,
                                                                           lowercase=opt.lowercase)
                output_lines.append(
                        ids[i] + "\t" + ' '.join(tokenizer_sen) + "\t" + tags[i] + "\n")
                fw.write(ids[i] + "\t" + ' '.join(tokenizer_sen) + "\t" + tags[i] + "\n")

