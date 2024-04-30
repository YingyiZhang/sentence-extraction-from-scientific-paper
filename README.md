# Extracting Problem and Method Sentence from Scientific Papers

## Overview

**Dataset and source code for paper "Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization".**

## Directory structure
> -   FE select: code and data for FE selection and data augmentation
>> -   FES: Data used for FE selection
>>> *   Iwatsuki-2021-EACL: The dataset provided by Iwatsuki & Aizawa (2021). The N-gram FE are selected from this dataset.
>>> *   train: The dataset used in getFEsPattern.py
>>> *   train_json: The dataset selected from SCIERC  (Luan et al., 2018). The dataset is used in getFEsPattern.py
>>> *   FES_method: N-gram FEs related to the method
>>> *   FES_problem: N-gram FEs related to the problem
>>> *   select_from_semeval_train.txt: The data used in getFEsPattern.py

>> -   model: fold used to store TFIDF model
>> -   acl_annoAugment.py: Code used for data augmentation. The function of unigram FE selection is in this code.  
>> -   getFEsPattern.py: Code used for selecting dependency pattern from sentences. The pre-step for unigram FE selection.
>> -   selectFromFES.py: Code used for selecting N-gram FE from dataset "Iwatsuki-2021-EACL"
>> -   tfidf_train.py: Code used for training TFIDF model for data augmentation

> -   sentence extraction: Code and data for sentence extraction
>> -   dataset: Dataset for training and testing
>> -   experiments: Parameter settings
>> -   results: Fold to store results
>> -   sen_tokenize.py: Code used for sentence and word tokenization
>> -   train_naive.py: Code to train the PLM-BiLSTM-ATT
>> -   train_trans.py: Code to train the PLM-BiLSTM-TRANS
>> -   train_sequ.py: Code to train the Sequential model
>> -   train_adjoin.py: Code to train the Concatenation model
>> -   train_context_enhanced.py: Code to train the context-enhanced transformer
> -   readme.txt: This file


## Dependency package:
1.  torch 1.10.0  
1.  nlpaug  1.1.10
1.  transformers  4.24.0
1.  treelib  1.6.1


## To train and test the model
use <code>python train_naive.py</code>

## To conduct data augmentation
use <code>python acl_annoAugment.py</code>
*   Change the augmenter in the code to select augmenters.
1.  --TFIDF Augmenter: tfidf
2.  --Embeding Augmenter: wordEmbed
3.  --Contextual embedding BERT Augmenter:contextWordEmbed
4.  --random Augmenter: random
5.  --back translation Augmenter: backTrans
6.  --unigramFE Augmenter:keywords
7.  --N-gramFE Augmenter:FEs

## To conduct unigram FE selection
use 
*  <code>python getFEsPattern.py</code>
*  <code>python acl_annoAugment.py</code> and select unigramFE Augmenter in the code


## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Yingyi Zhang, Chengzhi Zhang\*. Extracting Problem and Method Sentence from Scientific Papers: A Context-enhanced Transformer Using Formulaic Expression Desensitization. ***Scientometrics***, 2024. （in press)  [[doi]]()  [[Dataset & Source Code]](https://github.com/YingyiZhang/sentence-extraction-from-scientific-paper) 
