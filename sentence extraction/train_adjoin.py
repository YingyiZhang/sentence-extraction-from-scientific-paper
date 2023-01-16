"""Train and evaluate the model"""

import argparse
import random
import logging

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from model_adjoin import newBertForTokenClassification
from data_loader_adjoin import DataLoader
from evaluate_adjoin import evaluate
import utils

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TOKENIZER_NAMES = ['roberta-base', 'bert-base-uncased', 'facebook/bart-large', 'word',
                   'allenai/scibert_scivocab_uncased']
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=r'dataset/',
                    help="Directory containing the dataset")
parser.add_argument('--result_dir', default=r'results/', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='allenai/scibert_scivocab_uncased', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--langModel', default='scibert')
parser.add_argument('--model_dir',
                    default='experiments/semeval/scibert',
                    help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_dir',
                    default='experiments/semeval/scibert',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_"
                         "dir containing weights to reload before training")
parser.add_argument('--multi_gpu', default=True, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
parser.add_argument('--use_att', default=True, help="use attention mechnism")
parser.add_argument('--use_BiLSTM', default=True, help="use BiLSTM")
parser.add_argument('--classi', default=4, help="classification tag number")
parser.add_argument('--isadjoin', default=True, help="adjoin train")
parser.add_argument('--save_model', default=True, help="adjoin train")
parser.add_argument('--dataset', default="semeval", help="semeval or acl")
parser.add_argument('--num', default="0", help="fold")
parser.add_argument('--aug', default=None, help="fold")

def train(model, data_iterator, optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()
    scheduler.step()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(params.train_steps)
    for i in t:
        if params.isadjoin:
            batch_data, batch_data_before, batch_data_end, batch_tags, max_len, batch_lengths = next(data_iterator)
        else:
            batch_data, batch_tags, max_len, batch_lengths = next(data_iterator)
        batch_masks = batch_data.gt(0)

        if params.isadjoin:
            loss = model(batch_data, data_before=batch_data_before, data_end=batch_data_end, token_type_ids=None,
                         attention_mask=batch_masks, labels=batch_tags,
                         max_seq_len=max_len, input_lengths=batch_lengths, use_att=params.use_att,
                         use_BiLSTM=params.use_BiLSTM)
        else:
            loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags,
                         max_seq_len=max_len, input_lengths=batch_lengths, use_att=params.use_att,
                         use_BiLSTM=params.use_BiLSTM)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, model_dir, data_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.restore_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        if params.train_size % params.batch_size > 0:
            params.train_steps = (params.train_size // params.batch_size) + 1
        else:
            params.train_steps = (params.train_size // params.batch_size)
        if params.val_size % params.batch_size > 0:
            params.val_steps = (params.val_size // params.batch_size) + 1
        else:
            params.val_steps = (params.val_size // params.batch_size)

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        # Train for one epoch on training set
        train(model, train_data_iterator, optimizer, scheduler, params)

        # data iterator for evaluation
        # train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = params.train_steps
        params.eval_steps = params.val_steps
        val_metrics, pred_tags, pred_matrix, attention_outputs = evaluate(model, val_data_iterator, params, epoch, mark='Val')
        if len(val_metrics) > 0:
            val_f1 = val_metrics['Macro_F1']
            improve_f1 = val_f1 - best_val_f1

            # Save weights of the network
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            optimizer_to_save = optimizer.optimizer if args.fp16 else optimizer
            if params.save_model:
                utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model_to_save.state_dict(),
                                   'optim_dict': optimizer_to_save.state_dict()},
                                  is_best=improve_f1 > 0,
                                  checkpoint=model_dir+"/"+"adjoin"+str(params.num))
            if improve_f1 > 0:
                logging.info("- Found new best F1")
                best_val_f1 = val_f1
                fw_squence = open(params.result_dir, 'w')
                for tag, matrix in zip(pred_tags, pred_matrix):
                    fw_squence.write(tag + "\t" + matrix + "\n")

                fw_squence.close()

                #fw_squence_att = open(params.result_att_dir, 'w')

                #for i, matrix in enumerate(attention_outputs):
                #    for value in matrix:
                #        fw_squence_att.write(str(i)+"\t"+str(value))
                #        fw_squence_att.write("\n")
                #fw_squence_att.close()

                if improve_f1 < params.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping and logging best f1
            if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
                logging.info("Best val f1: {:05.2f}".format(best_val_f1))
                break


if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu
    params.use_att = args.use_att
    params.use_BiLSTM = args.use_BiLSTM
    params.classi = args.classi
    params.isadjoin = args.isadjoin
    params.dataset = args.dataset
    params.num = args.num
    params.save_model = args.save_model
    params.aug = args.aug


    params.result_dir = args.result_dir + "/"+params.dataset
    params.result_att_dir = args.result_dir + "/" + params.dataset + "_att"
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.squence.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(params.device, params.n_gpu, args.fp16))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)

    # Load training data and test data
    if params.aug is not None:
        train_data = data_loader.load_data(params.dataset+'/scibert/tenfold_new/combineAug/trainWithTrue_adjoin_'+
                                           params.aug+params.num, category='train')
    else:
        train_data = data_loader.load_data(params.dataset+'/scibert/adjoin_train'+params.num,
                                       category='train')
    val_data = data_loader.load_data(params.dataset+'/scibert/adjoin_test'+params.num)

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    # Prepare model
    model = newBertForTokenClassification.from_pretrained(args.bert_model_dir, num_labels=args.classi, is_adjoin= args.isadjoin)
    model.to(params.device)
    if args.fp16:
        model.half()

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[0,1])

    # Prepare optimizer
    if params.full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        'finetune change'
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("lease install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=params.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = Adam(optimizer_grouped_parameters, lr=params.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))

    train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, args.model_dir, args.data_dir,
                       args.restore_file)

