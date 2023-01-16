import os
os.environ["MODEL_DIR"] = 'model'
import sklearn.datasets
import re

import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


# Load sample dataset
# train_data = sklearn.datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# train_x = train_data.dataset
file_dir = u"acl"
tokenized = True

with open(file_dir,'r', errors='ignore', encoding='utf-8') as fr:
    lines = fr.readlines()
    # Tokenize input
    if tokenized == False:
        train_x_tokens = [_tokenizer(x.strip().split("\t")[1]) for x in lines]
    else:
        train_x_tokens = [x.strip().split("\t")[1].split() for x in lines]
fr.close()

#

# Train TF-IDF model
tfidf_model = nmw.TfIdf()
tfidf_model.train(train_x_tokens)
tfidf_model.save(os.environ["MODEL_DIR"])

# Load TF-IDF augmenter
aug = naw.TfIdfAug(model_path=os.environ["MODEL_DIR"], tokenizer=_tokenizer)

texts = [
    'The quick brown fox jumps over the lazy dog',
    'asdasd test apple dog asd asd'
]

for text in texts:
    augmented_text = aug.augment(text)

    print('-' * 20)
    print('Original Input:{}'.format(text))
    print('Agumented Output:{}'.format(augmented_text))