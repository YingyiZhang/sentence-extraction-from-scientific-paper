import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForTokenClassification, BertForTokenClassification, BartForSequenceClassification

import torch.nn.functional as F

'need change'
class newBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config, num_labels=4, is_adjoin=None):
        super(newBertForTokenClassification, self).__init__(config)
        self.embed_size = 768 #bart 1024 768
        self.hidden_size = 150
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                 bidirectional=True, batch_first=True)
        if is_adjoin:
            self.lstm_adjoin = nn.LSTM(input_size=self.embed_size*2, hidden_size=int(self.embed_size/2), num_layers=self.num_layers,
                                 bidirectional=True, batch_first=True)
            self.adjoin_liner = nn.Linear(self.embed_size*2, self.embed_size)


        self.init_weights()
        self.num_directions = 2
        self.decoding_rnn = nn.LSTMCell(input_size=self.hidden_size*2, hidden_size=self.hidden_size*2)
        self.num_labels = num_labels

        self.target_size = self.num_labels
        self.weights_layer = nn.Linear(self.hidden_size * 2 * self.num_layers, self.hidden_size * 2)

        self.classifier = nn.Linear(self.hidden_size * 2, self.target_size)
        self.act_func = nn.Softmax(dim=1)

        self.sig = nn.Sigmoid()

        'multi head'
        num_head = 6
        dim_model = 300

        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)

        self.fc1 = nn.Linear(dim_model, dim_model)
        self.fc2 = nn.Linear(dim_model, dim_model)

        self.attention_weights_layer = nn.Linear(self.hidden_size, self.hidden_size*2)

    def Position_wise_Feed_Forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

    def self_attention(self, Q, K, V, scale):
        '''
                Args:
                    Q: [batch_size, len_Q, dim_Q]
                    K: [batch_size, len_K, dim_K]
                    V: [batch_size, len_V, dim_V]
                    scale: 缩放因子 论文为根号dim_K
                Return:
                    self-attention后的张量，以及attention张量
                '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


    def Multi_Head_Attention(self, Q_ori, K_ori, V_ori):
        batch_size = Q_ori.size(0)

        Q = self.fc_Q(Q_ori)
        K = self.fc_K(K_ori)
        V = self.fc_V(V_ori)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5  # 缩放因子
        context, attention_output = self.self_attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + V_ori  # 残差连接
        out = self.layer_norm(out)
        return out, attention_output

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,  labels2=None, max_seq_len=None, input_lengths=None, use_att = False, use_att_adjoin=False, use_BiLSTM= True, layers=1):
        batch_size = len(input_ids)
        'need change'
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output, _ = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #sequence_output = self.dropout(sequence_output[0])#bart
        sequence_output = sequence_output[0]
        sequence_output  = self.dropout(sequence_output)#test1 test6 test7

        if use_BiLSTM:
            #if data_before is not None or data_end is not None:
            sequence_output, (sequence_hidden, cell) = self.lstm(sequence_output)
            if self.num_directions==2:
                # Optionally, Sum bidirectional RNN outputs
                sequence_output = torch.cat((sequence_output[:, :, :self.hidden_size], sequence_output[:, :, self.hidden_size:]), -1)
                logits = self.classifier(sequence_output[-1])
            if use_att:
                Q = sequence_output
                K = sequence_output
                V = sequence_output

                sequence_output, attention_output = self.Multi_Head_Attention(Q, K, V)
                sequence_output = self.Position_wise_Feed_Forward(sequence_output)
                sequence_output = torch.mean(sequence_output, 1)
                logits = self.classifier(sequence_output)
                logits = self.act_func(logits)

        loss_fct = CrossEntropyLoss()

        # Only keep active parts of the loss
        if labels is not None:
            loss_etc = loss_fct(logits, labels)
            return loss_etc
        else:
            return logits, attention_output