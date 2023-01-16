import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MODEL_DIR"] = 'model'
import nlpaug.augmenter.word as naw
import json
import random
from collections import Counter


class DataProcessor(object):
    """Base class for dataset converters for sequence classification dataset sets."""

    def get_train_examples(self, raw_data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, raw_data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this dataset set."""
        raise NotImplementedError()

    def get_train_size(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(line.strip())
            return lines

class TextClassProcessor(DataProcessor):

    def get_train_examples(self, raw_data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_tsv(raw_data_dir), "train")
        return examples

    def _create_examples(self, lines, set_type, skip_unsup=True,
                         only_unsup=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if skip_unsup and line[0] == "unsup":
                continue
            if only_unsup and line[0] != "unsup":
                continue

            sp = line.split("\t")
            guid = "%s-%d" % (set_type, int(sp[0]))
            if len(sp)==3:
                text_a = sp[1]
                text_b = None
                label = sp[2]
                text_a = text_a
                if text_b is not None:
                    text_b = text_b
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

            if len(sp)==6:
                text_a = sp[1]
                text_b = None
                pos = sp[2]
                den = sp[3]
                length = sp[4]
                label = sp[5]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos, den=den,length=length))
        return examples

class InputExample(object):
    """A single training/acl example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, pos=None, den=None, length=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for acl examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.pos = pos
        self.den = den
        self.length=length

def TFIDFAugmenter(text, action='substitute', aug_p = 0.2):
    aug = naw.TfIdfAug(model_path=os.environ.get('MODEL_DIR'), action=action, aug_p=aug_p)
    augmented_texts = aug.augment(text)
    return augmented_texts


def wordEmbedAugmenter(text, model_dir, embedding='glove.6B.300d.txt', action='substitute', aug_p=0.2, aug=None):
    "##### Substitute word by word2vec similarity"
    print (text)
    print ('finish')
    augmented_texts = aug.augment(text)
    return augmented_texts

def contextWordEmbedAugmenter(text, model_path='bert-base-uncased', action = 'substitute', aug_p = 0.3, aug=None):
    '''

    :param text: input text
    :param model_path:
        'bert-base-uncased',
        'bert-base-cased',
        'distilbert-base-uncased',
        'roberta-base',
        'distilroberta-base',
        'xlnet-base-cased',
    :param action: insert or 'substitute
    :param aug_p: 0.3/0.2
    :return: augmented text
    '''
    augmented_texts = aug.augment(text)  # 'distilbert-base-uncased'  'roberta-base'
    return augmented_texts

def randomWordAugmenter(text, action='substitute', aug_p=0.3):
    aug = naw.RandomWordAug(action=action, aug_p = aug_p)
    augmented_texts = aug.augment(text)
    return augmented_texts

def backTranslationAugmenter(text, aug=None):
    '''
    this funtion needs fastBPE
    :param text:
    :return:
    '''
    augmented_texts = aug.augment(text)
    return augmented_texts

def keywordsAugmenter(text, keywords_M, keywords_T, aug_p=0.2):
    tokens = text.strip().split()
    length = len(tokens)
    random.seed(2021)
    ids = []
    for i in range(0, length):
        ids.append(i)
    random.shuffle(ids)
    count = 0
    for id in ids:
        token = tokens[id]
        if count<length*0.2:
            if keywords_M.count(token.lower())>0:
                tokens[id]='_'
            if keywords_T.count(token.lower()) > 0:
                tokens[id] = '_'
            count = count+1
    line = ' '.join(tokens)
    return line

def FEAugmenter(text, keywords_M, keywords_T, aug_p=0.2):
    tokens = text.strip()
    length = len(tokens)
    random.seed(2021)
    ids = []
    for i in range(0, length):
        ids.append(i)
    random.shuffle(ids)
    count = 0

    for keyword_M in keywords_M:
        keyword_M  = keyword_M.strip()
        if count < 2:
            if keyword_M.count("*")>0:
                sp = keyword_M.split("*")
                if tokens.lower().count(sp[0].lower()) > 0 and tokens.lower().count(sp[1].lower()) > 0:
                    tokens = tokens.lower().replace(sp[0].lower(), "_")
                    tokens = tokens.lower().replace(sp[1].lower(), "_")
                    count = count + 1
            else:
                if tokens.lower().count(keyword_M.lower())>0:
                    tokens = tokens.lower().replace(keyword_M.lower(), "_")
                    count = count + 1

    for keyword_T in keywords_T:
        keyword_T = keyword_T.strip()
        if count < 2:
            if keyword_T.count("*")>0:
                sp = keyword_T.split("*")
                if tokens.lower().count(sp[0].lower()) > 0 and tokens.lower().count(sp[1].lower()) > 0:
                    tokens = tokens.lower().replace(sp[0].lower(), "_")
                    tokens = tokens.replace(sp[1].lower(), "_")
                    count = count + 1
            else:
                if tokens.lower().count(keyword_T.lower())>0:
                    tokens = tokens.lower().replace(keyword_T.lower(), "_")
                    count = count + 1
    return tokens

def readKeywords(lines):
    '''
        将保存的diction形式的tree，转化为tree的形式
        :return:
        '''
    def _iter(nodes, parent_id):
        for k, v in nodes.items():
            children = v.get('children', None)
            data = v.get('data', None)
            if children:
                yield (k, data, parent_id)
                for child in children:
                    for i in _iter(child, k):
                        yield i
            else:
                yield (k, data, parent_id)

    need_keywords = []
    for line in lines:
        for i in _iter(json.loads(line), None):
            word = i[0]
            data = i[1]
            if data=='None':
                need_keywords.append(word)
            if word.count("{+METHOD}")>0 or word.count("{+TASK}")>0:
                if word.count("{+METHOD}")>0:
                    need_keywords.append(word.replace("{+METHOD}",''))
                if word.count("{+TASK}")>0:
                    need_keywords.append(word.replace("{+TASK}", ''))
                if data.count(":")>0:
                    conj = data.split(":")[1].strip()
                    if conj!='and':
                        need_keywords.append(conj)
    return need_keywords

def word_level_augment(
        examples, aug_ops, action, model_dir,
        embedding='glove.6B.300d.txt', model_path='bert-base-uncased', aug=None, fw=None):
    """Word level augmentations. Used before augmentation."""
    print ('aug_start')
    if aug_ops:
        if aug_ops.startswith("random"):
            token_prob = float(aug_ops.split("-")[1])
            for i in range(len(examples)):
                examples[i].text_a = randomWordAugmenter(examples[i].text_a, aug_p=token_prob, action=action)
                aug_text = examples[i].text_a
                tag = examples[i].label

                if isinstance(aug_text, str):
                    fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")
                else:
                    fw.write(str(i + 1) + "\t" + str(" ".join(aug_text)) + "\t" + tag + "\n")

        if aug_ops.startswith("tfidf"):
            token_prob = float(aug_ops.split("-")[1])
            for i in range(len(examples)):
                examples[i].text_a = TFIDFAugmenter(examples[i].text_a, aug_p=token_prob, action=action)
                aug_text = examples[i].text_a
                tag = examples[i].label

                if isinstance(aug_text, str):
                    fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")
                else:
                    fw.write(str(i + 1) + "\t" + str(" ".join(aug_text)) + "\t" + tag + "\n")

        if aug_ops.startswith("wordEmbed"):
            "# model_type: word2vec, glove or fasttext"
            model_dir = model_dir
            token_prob = float(aug_ops.split("-")[1])
            for i in range(len(examples)):
                print (i)
                examples[i].text_a = wordEmbedAugmenter(examples[i].text_a, model_dir, embedding=embedding,
                                                 action=action, aug_p=token_prob, aug=aug)
                aug_text = examples[i].text_a
                tag = examples[i].label

                if isinstance(aug_text, str):
                    fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")
                else:
                    fw.write(str(i + 1) + "\t" + str(" ".join(aug_text)) + "\t" + tag + "\n")

        if aug_ops.startswith("contextWordEmbed"):
            token_prob = float(aug_ops.split("-")[1])
            for i in range(len(examples)):
                print (i)
                examples[i].text_a = contextWordEmbedAugmenter(examples[i].text_a, model_path=model_path,
                                                        action=action, aug_p=token_prob, aug=aug)
                aug_text = examples[i].text_a
                tag = examples[i].label
                if isinstance(aug_text, str):
                    fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")
                else:
                    fw.write(str(i + 1) + "\t" + str(" ".join(aug_text)) + "\t" + tag + "\n")

        if aug_ops.startswith("backTrans"):
            for i in range(len(examples)):
                print (i)
                if i>=0:
                    examples[i].text_a = backTranslationAugmenter(examples[i].text_a, aug=aug)
                    aug_text = examples[i].text_a
                    tag = examples[i].label
                    print (aug_text)
                    if isinstance(aug_text, str):
                        fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")
                    else:
                        fw.write(str(i + 1) + "\t" + str(" ".join(aug_text)) + "\t" + tag + "\n")

        if aug_ops.startswith("keywords"):
            with open(
                    "FES/seed_patternz_M",
                    'r') as fr:
                needKeywords_M = readKeywords(fr)

            with open(
                        "FES/seed_patternz_T",
                        'r') as fr:
                needKeywords_T = readKeywords(fr)
            token_prob = float(aug_ops.split("-")[1])

            for i in range(len(examples)):
                examples[i].text_a = keywordsAugmenter(examples[i].text_a, needKeywords_M, needKeywords_T, token_prob)
                aug_text = examples[i].text_a
                tag = examples[i].label

                fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")

        if aug_ops.startswith("FEs"):
            with open(
                    "FES/FES_method",
                    'r') as fr:
                needFE_M = fr.readlines()

            with open(
                        "FES/FES_problem",
                        'r') as fr:
                needFE_T = fr.readlines()

            token_prob = float(aug_ops.split("-")[1])
            for i in range(len(examples)):
                print (i)
                sen = examples[i].text_a
                examples[i].text_a = FEAugmenter(examples[i].text_a, needFE_M, needFE_T, token_prob)
                aug_text = examples[i].text_a
                tag = examples[i].label

                if isinstance(aug_text, str):
                    fw.write(str(i + 1) + "\t" + aug_text + "\t" + tag + "\n")
                else:
                    fw.write(str(i + 1) + "\t" + str(" ".join(aug_text)) + "\t" + tag + "\n")

                if examples[i].text_a!=sen:
                    print ('true', examples[i].text_a)
    return examples



'''
aug_ops:
TFIDFAugmenter: tfidf
wordEmbedingAugmenter: wordEmbed
ContextualEmbeddingBERTAugmenter Augmenter:contextWordEmbed
randomWordAugmenter: random
backTranslationAugmenter: backTrans
unigramFEAugmenter:keywords
N-gramFEAugmenter:FEs
'''

option = ['random',"tfidf","wordEmbed","contextWordEmbed","backTrans", 'keywords', "FEs"]
augmenter = "keywords"
daset = 'acl_anno'
action = 'substitute'
model_dir = ''
aug_ops = augmenter+"-0.2-new"

if augmenter == 'wordEmbed':
    aug = naw.WordEmbsAug(model_type='glove', model_path=os.path.join(model_dir, "glove.6B.300d.txt"),
                     action=action, aug_p=0.2)
if augmenter == 'contextWordEmbed':
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action=action, aug_p=0.2)
if augmenter == 'backTrans':
    aug = naw.BackTranslationAug(device='cuda')

for number in range(0,1):
    raw_data_dir = u"../neuralModel/hugFace/dataset/acl/scibert/tenfold_new/train0"
    processor = TextClassProcessor()

    examples = processor.get_train_examples(raw_data_dir)
    print ('examples down')
    print (len(examples))

    with open(os.path.join("../neuralModel/hugFace/dataset/acl/scibert/tenfold_new", aug_ops.split("-")[0], 'train0'),"w") as fw:
        examples = word_level_augment(examples, aug_ops, action, model_dir='',
            embedding='glove.6B.300d.txt', model_path='bert-base-uncased', aug=None, fw=fw)

    #fw.close()