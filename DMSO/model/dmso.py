import torch
from torch import nn
import math
import torch.nn.functional as F
from fastNLP.embeddings.embedding import TokenEmbedding, Embedding


class DMSO(nn.Module):
    def __init__(self, embed, hidden_size, num_classes=4, dropout=0.2):
        super(DMSO, self).__init__()
        if isinstance(embed, TokenEmbedding) or isinstance(embed, Embedding):
            self.embedding = embed
        else:
            self.embedding = Embedding(embed)

        self.encoder = WordEncoder(self.embedding.embed_size, hidden_size)
        self.loc = LocalInteractionLayer(hidden_size*2, hidden_size)
        self.glo = GlobalInteractionFusionLayer(hidden_size*2, hidden_size)
        self.prediction = Prediction(hidden_size, num_classes, dropout)


    def forward(self, words1, words2):
        a_emb = self.embedding(words1)
        b_emb = self.embedding(words2)
        a_emb = a_emb.view(words1.size(0),-1,a_emb.size(-1))
        b_emb = b_emb.view(words2.size(0),-1,b_emb.size(-1))
        mask_wa = torch.ne(a_emb, 0)[:,:,0].unsqueeze(2)
        mask_wb = torch.ne(b_emb, 0)[:,:,0].unsqueeze(2)

        a_enc = self.encoder(a_emb)
        b_enc = self.encoder(b_emb)

        a_fus, b_fus = self.loc(a_enc, b_enc, mask_wa, mask_wb)

        x = self.glo(a_fus, b_fus, words1.size(), words2.size())

        return self.prediction(x)


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features, bias)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        if bias:
            nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)


class WordEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WordEncoder, self).__init__()
        self.enc = torch.nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
    def forward(self, a):
        self.enc.flatten_parameters()
        vec, _ = self.enc(a)
        return vec


class LocalInteractionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LocalInteractionLayer, self).__init__()
        self.match = IFE(input_size)

    def forward(self, a, b, mask_a, mask_b):
        a_mac, b_mac = self.match(a, b, mask_a, mask_b)
        return a_mac, b_mac
    

class SentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentEncoder, self).__init__()
        self.w = nn.Parameter(torch.tensor(1 / math.sqrt(input_size)))
        self.attn = nn.Linear(input_size, hidden_size) 
        self.sent = nn.Linear(hidden_size, 1, bias=False) 

    def forward(self, a_fus, words1):
        a = a_fus.view(-1,words1[2],a_fus.size(-1))
        attn = torch.tanh(self.attn(a))
        attn = self.sent(attn)
        sent = torch.matmul(F.softmax(attn, dim=1).transpose(1, 2),a)
        return sent.view(words1[0],-1,a_fus.size(-1))


class GlobalInteractionFusionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GlobalInteractionFusionLayer, self).__init__()
        self.enc = SentEncoder(input_size, hidden_size)
        self.match = IFE(input_size)
        self.pooling = Pooling()
        self.fusion = nn.Sequential(
            nn.Dropout(0.2),
            Linear(hidden_size* 4*2, hidden_size, activations=True),
        )

    def forward(self, a_fus, b_fus, words1, words2):
        a_sent = self.enc(a_fus, words1)
        b_sent = self.enc(b_fus, words2)
        mask_sa = torch.ne(a_sent, 0)[:,:,0].unsqueeze(2)
        mask_sb = torch.ne(b_sent, 0)[:,:,0].unsqueeze(2)
        a_mac, b_mac = self.match(a_sent, b_sent, mask_sa, mask_sb)
        
        a = self.pooling(a_mac, mask_sa)
        b = self.pooling(b_mac, mask_sb)
        return self.fusion(torch.cat([a, b, a-b, a*b], dim=-1))


class IFE(nn.Module):
    def __init__(self, input_size):
        super(IFE, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1 / math.sqrt(input_size)))
        self.beta = nn.Parameter(torch.tensor(1 / math.sqrt(input_size)))

    def normal_attn(self, a, b, w, mask_a, mask_b):
        attn = torch.matmul(a, b.transpose(1, 2)) * w
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).bool()
        attn.masked_fill_(~mask, -1e7)
        return attn
    
    def diff_attn(self, a, b, w, mask_a, mask_b):
        attn = torch.cdist(a,b, p=1) * (-w)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).bool()
        attn.masked_fill_(~mask, -1e7)
        return torch.sigmoid(attn)

    def forward(self, a, b, mask_a, mask_b):
        normal = self.normal_attn(a, b, self.alpha, mask_a, mask_b)
        diff = self.diff_attn(a, b, self.beta, mask_a, mask_b)
        attn = normal * diff
        a_mac = torch.matmul(F.softmax(attn, dim=2), b)
        b_mac = torch.matmul(F.softmax(attn, dim=1).transpose(1, 2), a)
        return a_mac, b_mac
    

class Pooling(nn.Module):
    def forward(self, x, mask):
        mask = mask.bool()
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]


class Prediction(nn.Module):
    def __init__(self, hidden_size, num_classes=4, dropout=0.2):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.dense(x) #


