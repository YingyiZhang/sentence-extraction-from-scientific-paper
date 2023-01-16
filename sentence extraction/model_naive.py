import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForTokenClassification, BertForTokenClassification, BartForSequenceClassification

import torch.nn.functional as F


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector: torch.Tensor, mask: torch.Tensor, dim: int,
               keepdim: bool = False,
               min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index

'need change'
class newBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config, num_labels=4):
        super(newBertForTokenClassification, self).__init__(config)
        self.embed_size = 768
        self.hidden_size = 150
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 bidirectional=True, batch_first=True)
        self.init_weights()
        self.num_directions = 2
        self.decoding_rnn = nn.LSTMCell(input_size=self.hidden_size*2, hidden_size=self.hidden_size*2)
        self.num_labels = num_labels
        self.weight = torch.ones(self.hidden_size*2)

        self.target_size = self.num_labels

        self.classifier = nn.Linear(self.hidden_size * 2, self.target_size)
        self.act_func = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()


    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output  # [batch, seq_len, hidden_size]

        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        #print (weights)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2), weights


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, use_att=False, use_BiLSTM=True, input_lengths=None, max_seq_len=None):
        batch_size = len(input_ids)
        'need change'
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output, _ = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output = sequence_output[0]
        sequence_output = self.dropout(sequence_output[0])#bert roberta

        #sequence_output = self.dropout(sequence_output)

        if use_BiLSTM:
            sequence_output, (sequence_hidden, cell) = self.lstm(sequence_output)
            if self.num_directions==2:
                # Optionally, Sum bidirectional RNN outputs
                sequence_output = torch.cat((sequence_output[:, :, :self.hidden_size], sequence_output[:, :, self.hidden_size:]), -1)
                logits = self.classifier(sequence_output[-1])
            if use_att:
                attn_output, weights = self.attention(sequence_output, sequence_hidden)
                logits = self.classifier(attn_output)
                logits = self.act_func(logits)

        loss_fct = CrossEntropyLoss()

        # Only keep active parts of the loss
        if labels is not None:
            loss_etc = loss_fct(logits, labels)
            return loss_etc
        else:
            return logits, weights