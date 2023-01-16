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
        self.pos_size = pos_size
        self.den_size = den_size
        self.len_size = len_size
        if self.pos_size is not None:
            print('pos_size', pos_size)
            self.pos_embedding = nn.Embedding(pos_size, 20)
        if self.den_size is not None:
            print('den_size', den_size)
            self.den_embedding = nn.Embedding(den_size, 20)
        if self.len_size is not None:
            print ('len_size', len_size)
            self.len_embedding = nn.Embedding(len_size, 20)

        self.lstm = nn.LSTM(input_size=self.embed_size*2, hidden_size=self.hidden_size, num_layers=1,
                                 bidirectional=True, batch_first=True)
        if is_adjoin:
            self.lstm_adjoin = nn.LSTM(input_size=self.embed_size*2, hidden_size=20, num_layers=1,
                                 bidirectional=True, batch_first=True)

        self.init_weights()
        self.num_layers = 1
        self.num_directions = 2
        self.decoding_rnn = nn.LSTMCell(input_size=self.hidden_size*2, hidden_size=self.hidden_size*2)
        self.num_labels = num_labels
        self.target_size = self.num_labels

        self.classifier = nn.Linear(self.hidden_size * 2, self.target_size)
        self.act_func = nn.Softmax(dim=1)

        self.adjoin_liner = nn.Linear(self.embed_size * 2, self.embed_size)

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output  # [batch, seq_len, hidden_size]

        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)

        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, input_ids, data_before=None, data_end=None, poss_ids=None, dens_ids=None, length_ids=None, token_type_ids=None, attention_mask=None, labels=None,  labels2=None, max_seq_len=None, input_lengths=None, use_att = False, use_att_adjoin=False, use_BiLSTM= True, use_cnn = False, layers=1):
        batch_size = len(input_ids)
        'need change'
        if data_before is not None:
            sequence_before_output = self.bert(data_before, token_type_ids=token_type_ids, attention_mask=attention_mask)
            sequence_before_output = self.dropout(sequence_before_output[0])
            sequence_adjoin_ouput = sequence_before_output
        if data_end is not None:
            sequence_end_output = self.bert(data_end, token_type_ids=token_type_ids,
                                                 attention_mask=attention_mask)
            sequence_end_output = self.dropout(sequence_end_output[0])
            if data_before is not None:
                sequence_adjoin_ouput = torch.cat((sequence_before_output, sequence_end_output), -1)
            else:
                sequence_adjoin_ouput = sequence_end_output
        if data_before is not None or data_end is not None:
            attn_adjoin_output = self.adjoin_liner(sequence_adjoin_ouput)

        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output, _ = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output = self.dropout(sequence_output[0])
        sequence_output = sequence_output[0]

        if use_BiLSTM:
            if data_before is not None or data_end is not None:
                sequence_output = torch.cat((sequence_output, attn_adjoin_output), -1)
            sequence_output, (sequence_hidden, cell) = self.lstm(sequence_output)
            if self.num_directions==2:
                # Optionally, Sum bidirectional RNN outputs
                sequence_output = torch.cat((sequence_output[:, :, :self.hidden_size], sequence_output[:, :, self.hidden_size:]), -1)
                logits = self.classifier(sequence_output[-1])
            if use_att:
                attn_output = self.attention(sequence_output, sequence_hidden)
                logits = self.classifier(attn_output)
                logits = self.act_func(logits)

        loss_fct = CrossEntropyLoss()

        # Only keep active parts of the loss
        if labels is not None:
            loss_etc = loss_fct(logits, labels)
            return loss_etc
        else:
            return logits