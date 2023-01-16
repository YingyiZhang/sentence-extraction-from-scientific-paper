"""Evaluate the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

from metrics import f1_score

from data_loader_sequ import DataLoader
import utils
from model_hire import newBertForTokenClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TOKENIZER_NAMES = ['roberta-base', 'bert-base-uncased', 'facebook/bart-large', 'word', 'allenai/scibert_scivocab_uncased']
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=r'dataset', help="Directory containing the dataset")
parser.add_argument('--result_dir', default=r'results/scibert', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='allenai/scibert_scivocab_uncased', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='experiments/semeval/scibert', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_file', default='best', help="name of the file in `model_dir` containing weights to load")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--beam', default=1, help="how many keyphrase to extract")
parser.add_argument('--use_att', default=True, help="use attention mechnism")
parser.add_argument('--use_BiLSTM', default=True, help="use BiLSTM")
parser.add_argument('--classi', default=4, help="classification tag number")
parser.add_argument('--isadjoin', default=True, help="adjoin train")
#tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
#tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)


def evaluate(model, data_iterator, params, epoch, mark='Eval', verbose=False, fw_squence=None):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    classi = params.classi
    idx2tag = params.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    for round in range(params.eval_steps):
        # fetch the next evaluation batch
        if params.isadjoin:
            batch_data, batch_data_before, batch_data_end, \
            batch_tags, batch_tags_before, batch_tags_end,\
            max_len, batch_lengths = next(
                data_iterator)
        else:
            batch_data, batch_tags, max_len, batch_lengths = next(data_iterator)
        batch_masks = batch_data.gt(0)

        if params.isadjoin:
            batch_output = model(batch_data, data_before=batch_data_before, data_end=batch_data_end,
                                 token_type_ids=None, attention_mask=batch_masks, max_seq_len=max_len,
                                 input_lengths=batch_lengths, use_att=params.use_att, use_BiLSTM=params.use_BiLSTM)
        else:
            batch_output = model(batch_data,
                                 token_type_ids=None, attention_mask=batch_masks, max_seq_len=max_len,
                                 input_lengths=batch_lengths, use_att=params.use_att, use_BiLSTM=params.use_BiLSTM)  # shape: (batch_size, max_len, num_labels)

        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()


        # for batch_out, batch_tag in zip(batch_output, batch_tags):
        #     fw_squence.write('score'+"\t")
        #     for out in batch_out:
        #         fw_squence.write(str(out)+"\t")
        #     fw_squence.write("\n")
        #     batch_out = np.argmax(batch_out)
        #     fw_squence.write(str(batch_tag) + "\n")
        #     fw_squence.write(str(batch_out) + "\n")

        pred_matrix=[idx2tag.get(np.argmax(batch[1])) for batch in batch_output]
        pred_tags.extend(pred_matrix)
        true_tags.extend(['O' if idx2tag.get(idx) is None else idx2tag.get(idx) for idx in batch_tags])
        assert len(pred_tags) == len(true_tags)


    # logging loss, f1 and report
    metrics = {}
    #need change
    print('true_tags', true_tags)
    print('pred_tags', pred_tags)
    f1 = f1_score(true_tags, pred_tags, classi = params.classi)
    print('f1_score', f1)

    metrics['Macro_F1'] = f1["Macro_F1"]
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    return metrics, pred_tags, pred_matrix


if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available()  else "cpu")
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu
    params.use_att = args.use_att
    params.use_BiLSTM = args.use_BiLSTM
    params.classi = args.classi
    params.isadjoin = args.isadjoin

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)

    # Load data
    test_data = data_loader.load_data('semeval/scibert/keywords-new_adjoin')

    # Specify the test set size
    params.test_size = test_data['size']
    params.eval_steps = params.test_size // params.batch_size+1
    print (params.test_size, params.batch_size)
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

    logging.info("- done.")


    # Prepare model
    model = newBertForTokenClassification.from_pretrained(args.bert_model_dir, num_labels=args.classi,
                                                          is_adjoin=args.isadjoin)
    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    if args.fp16:
        model.half()
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)
    logging.info("Starting evaluation...")
    fw_squence=open(args.result_dir+"/adjoin","w")
    epoch=0
    test_metrics, pred_tags, pred_matrix = evaluate(model, test_data_iterator, params, epoch, mark='Eval', verbose=True)
    for tag, matrix in zip( pred_tags, pred_matrix):
        fw_squence.write(tag+"\t"+matrix+"\n")
    fw_squence.close()