import torch
import torch.nn as nn
import math
from torch import Tensor

from seq2seq.data_loader import en_total_words, cn_total_words, train_dataloader, test_dataloader, cn_bos_idx, \
    cn_eos_idx, decode_sents, PAD_IDX


def generate_square_subsequent_mask(sz):
    """防止一次读入全部目标序列，一次多读一个字符 ， 为-inf会被忽略（float tensor视为权重）"""
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # False：不忽略，True：忽略
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    # 标记pad字符，为true的是pad，将会被忽略
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):

    def __init__(self, emb_size: int, max_len: int = 5000, dropout: float = 0.2):
        super(PositionalEncoding, self).__init__()
        # Position Encoding. Transformer位置无关，需要加上位置编码。 i in [0,emb_size/2)
        # PE(pos,2i) = sin(pos/10000^(2i/d)) # 偶数位置
        # PE(pos,2i+1) = cos(pos/10000^(2i/d)) # 奇数位置
        # 对 pos/10000^(2i/d) 取log就是下面的东西
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_emb = torch.zeros((max_len, emb_size))
        pos_emb[:, 0::2] = torch.sin(pos * den)
        pos_emb[:, 1::2] = torch.cos(pos * den)
        pos_emb = pos_emb.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, token_emb: Tensor):
        return self.dropout(token_emb + self.pos_emb[:token_emb.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # embedding 是分布是N(0,1)，乘上math.sqrt(self.emb_size)，增加var
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2Seq(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size,
                 tgt_vocab_size, dim_feedforward: int = 512, dropout: float = 0.2):
        super(Seq2Seq, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_token_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_token_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_token_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_token_emb(tgt))

        # memory_mask设置为None,不需要mask; memory=encoder(input)
        # memory_key_padding_mask 和 src_padding_mask 一样，
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_token_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_token_emb(tgt)), memory, tgt_mask)

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


EMB_SIZE = 256
model = Seq2Seq(3, 3, emb_size=EMB_SIZE, nhead=8, src_vocab_size=en_total_words, tgt_vocab_size=cn_total_words,
                dim_feedforward=EMB_SIZE)
model.init()
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def train_epoch():
    model.train()
    losses = 0
    for i, (src, src_lens, tgt, tgt_lens) in enumerate(train_dataloader):
        # 转换为 word_index, batch_size。可以将一个批量的句子，视为一个句子处理。
        # batch_size和embedding_size是固定的，句子长度不固定，把句子长度做个第一个维度方便处理。
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # 去掉最后一个字符, input:[BOS,w1,w2] -> output:[w1,w2,BOS]
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        # 去掉第一个字符BOS，因为模型的输出是BOS之后的字符BOS->第一个字符...
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        if i % 200 == 0:
            print("loss:", loss.item())
            test_translate()

    return losses / len(train_dataloader)


def translate(src, src_mask, max_len=20):
    memory = model.encode(src, src_mask)
    batch_size = src.size(1)
    ys = torch.ones(1, batch_size).fill_(cn_bos_idx).type(torch.long)
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
    return ys


@torch.no_grad()
def test_translate():
    model.eval()
    for i, (src, src_lens, tgt, tgt_lens) in enumerate(test_dataloader):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        pred = translate(src, src_mask)
        print(decode_sents(src.transpose(0, 1), False))
        print(decode_sents(tgt.transpose(0, 1)))
        print(decode_sents(pred.transpose(0, 1)))
        break


if __name__ == '__main__':
    for i in range(10):
        loss = train_epoch()
        print(f"epoch {i} loss: {loss}")
        torch.save(model.state_dict(), f"transformer_epoch_{i}.model")
