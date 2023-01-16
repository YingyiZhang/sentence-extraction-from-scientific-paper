import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForTokenClassification, BertForTokenClassification, BartForSequenceClassification

import torch.nn.functional as F

'need change'
class newBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config, num_labels=4, pos_size=None, den_size=None, len_size=None, is_adjoin=None):
        super(newBertForTokenClassification, self).__init__(config)
        self.embed_size = 768
        self.hidden_size = 150

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=1,
                                 bidirectional=True, batch_first=True)

        self.lstm_adjoin = nn.LSTM(input_size=self.hidden_size*2, hidden_size=self.hidden_size, num_layers=1,
                                   bidirectional=True, batch_first=True)

        self.init_weights()
        self.num_layers = 1
        self.num_directions = 2
        self.decoding_rnn = nn.LSTMCell(input_size=self.hidden_size*2, hidden_size=self.hidden_size*2)
        self.num_labels = num_labels

        self.target_size = self.num_labels

        self.classifier = nn.Linear(self.hidden_size * 2, self.target_size)
        self.act_func = nn.Softmax(dim=1)

        'initialize CNN'
        self.num_filters = 100
        self.kernel_sizes = [3, 4, 5]
        self.convs_1d = nn.ModuleList([nn.Conv2d(1, self.num_filters, (k, self.embed_size),
                                                 padding=(k-2, 0)) for k in self.kernel_sizes])

        self.fc = nn.Linear(len(self.kernel_sizes)*self.num_filters, self.num_labels)
        self.sig = nn.Sigmoid()

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output

        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)

        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, input_ids, data_before=None, data_end=None,
                token_type_ids=None, attention_mask=None, labels=None, labels_before=None, labels_end=None,
                labels2=None, max_seq_len=None, input_lengths=None,
                use_att = False, use_att_adjoin=False, use_BiLSTM= True, use_cnn = False, layers=1):
        batch_size = len(input_ids)
        'need change'
        if data_before is not None:
            sequence_before_output = self.bert(data_before, token_type_ids=token_type_ids, attention_mask=attention_mask)
            sequence_before_output = self.dropout(sequence_before_output[0])
        if data_end is not None:
            sequence_end_output = self.bert(data_end, token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask)
            sequence_end_output = self.dropout(sequence_end_output[0])

        if data_before is not None or data_end is not None:
            sequence_before_output, (sequence_before_hidden, before_cell) = self.lstm(sequence_before_output)
            if self.num_directions == 2:
                sequence_before_output = torch.cat(
                    (sequence_before_output[:, :, :self.hidden_size], sequence_before_output[:, :, self.hidden_size:]), -1)
            att_before_output = self.attention(sequence_before_output, sequence_before_hidden)

            sequence_end_output, (sequence_end_hidden, end_cell) = self.lstm(sequence_end_output)
            if self.num_directions == 2:
                sequence_end_output = torch.cat(
                    (sequence_end_output[:, :, :self.hidden_size], sequence_end_output[:, :, self.hidden_size:]), -1)
            attn_end_output = self.attention(sequence_end_output, sequence_end_hidden)
            if labels is not None:
                labels = torch.stack([labels, labels_before, labels_end], axis=1)


        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output = self.dropout(sequence_output[0])
        sequence_output = sequence_output[0]

        sequence_output, (sequence_hidden, cell) = self.lstm(sequence_output)
        if self.num_directions == 2:
            sequence_output = torch.cat(
                (sequence_output[:, :, :self.hidden_size], sequence_output[:, :, self.hidden_size:]), -1)
        attn_output = self.attention(sequence_output, sequence_hidden)

        if use_BiLSTM:
            if data_before is not None or data_end is not None:
                attn_output = torch.stack([att_before_output, attn_output, attn_end_output], axis=1)

            sequence_output, (sequence_hidden, cell) = self.lstm_adjoin(attn_output)
            if self.num_directions==2:
                # Optionally, Sum bidirectional RNN outputs
                sequence_output = torch.cat((sequence_output[:, :, :self.hidden_size], sequence_output[:, :, self.hidden_size:]), -1)
                logits = self.classifier(sequence_output)

        loss_fct = CrossEntropyLoss()
        if labels is not None:
            logits = logits.view(-1, self.target_size)
            labels = labels.view(-1)
            loss_etc = loss_fct(logits, labels)
            return loss_etc
        else:
            return logits