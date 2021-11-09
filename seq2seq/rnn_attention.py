import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Paper: https://arxiv.org/pdf/1508.04025

class Encoder(nn.Module):
    """
    编码器，非常简单，双向RNN,考虑前后上下文
    """

    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        x = self.dropout(self.embed(x))
        # 句子长度不一样，需使用pack_padded_sequence，告诉rnn，长度之后pad不需要处理
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, hid = self.rnn(x)
        # -> batch_size, x_len,2*hidden_size
        out, _ = pad_packed_sequence(out, batch_first=True)
        # hid dim: 2, batch_size, hidden_size -> batch_size,hidden_size*2
        hid = torch.cat((hid[-2], hid[-1]), dim=1)
        # hid匹配到dec_hidden_size,作为encoder hid 的输入
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)
        return out, hid


class Attention(nn.Module):
    """
    att = softmax(<encoded_ys,encoded_xs>)
    att_xs = att * encoded_xs
    output = [att_xs,ys]
    """

    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.fc_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.fc_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, xs, ys, mask):
        """
        1. 转换维度: xs' = fc(xs)
        2. 计算Att系数：att = softmax(<ys,xs'>)
        3. 获得Att: att_xs = att * xs
        4. 拼接：cat(att_xs,ys)
        :param xs: encoded inputs.dim: batch_size, x_len, x_size(encode_hidden_size)
        :param ys: encoded labels.dim: batch_size, 1 or y_len, y_size(decode_hidden_size)
        :param mask: PAD 字符为True,其他False
        :return: dim: batch_size,y_len,y_size
        """
        batch_size = xs.size(0)
        input_len = xs.size(1)
        output_len = ys.size(1)
        # 1.转换维度: 使得输入的维度，和输出维度匹配，否则没法做内积。 batch_size , x_len, x_size -> batch_size , x_len, y_size
        xs_ = self.fc_in(xs.view(batch_size * input_len, -1)).view(batch_size, input_len, -1)
        # 2.计算Att系数：x,y直接做内积获得x,y的相关性 ，
        # att = <x,y> = Y*X^T in batch -> batch_size , y_len , x_len(每个字符上的权重）
        att = torch.bmm(ys, xs_.transpose(1, 2))
        # PAD 字符的系数应趋于0
        att.masked_fill_(mask, -1e6)
        att = F.softmax(att, dim=2)
        # 3. 获得Att: ->batch_size,y_len,y_size
        att_xs = torch.bmm(att, xs)
        # 4. 拼接：cat(att_xs,ys) -> batch_size, y_len, x_size+y+size
        output = torch.cat((att_xs, ys), dim=2)
        # 下面两步：转换到输出维度 -> batch_size,y_len,y_size
        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.fc_out(output)).view(batch_size, output_len, -1)
        return output, att


class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, dec_hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_fc = nn.Linear(dec_hidden_size, vocab_size)

    def create_mask(self, ys_len, xs_len):
        """
        和输入字符无关的字符(PAD)标记为0，否则标记为1
        :param ys_len:
        :param xs_len:
        :return: (batch_size, ys_len, xs_len)
        """
        y_max = ys_len.max()
        x_max = xs_len.max()
        # 有效字符为true
        y_mask = ys_len[:, None] > torch.arange(y_max)[None, :]
        x_mask = xs_len[:, None] > torch.arange(x_max)[None, :]
        # true: y字符存在并且x字符也存在 , 然后取反
        return ~(y_mask[:, :, None] * x_mask[:, None, :])

    def forward(self, xs, xs_len, ys, ys_len, hid):
        ys = self.dropout(self.embed(ys))

        ys = pack_padded_sequence(ys, ys_len, batch_first=True, enforce_sorted=False)
        out, hid = self.rnn(ys, hid)
        out, _ = pad_packed_sequence(out, batch_first=True)

        mask = self.create_mask(ys_len, xs_len)
        out, att = self.attention(xs, out, mask)

        out = F.log_softmax(self.out_fc(out), -1)
        return out, hid, att


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, xs_len, ys, ys_len):
        enc_xs, hid = self.encoder(xs, xs_len)
        output, hid, att = self.decoder(enc_xs, xs_len, ys, ys_len, hid)
        return output, att

    def translate(self, xs, xs_len, ys, max_length=20):
        enc_xs, hid = self.encoder(xs, xs_len)

        preds = []
        batch_size = xs.size(0)
        atts = []
        for i in range(max_length):
            ys_len = torch.ones(batch_size)
            output, hid, att = self.decoder(enc_xs, xs_len, ys, ys_len, hid)
            ys = output.max(2)[1].view(batch_size, 1)
            preds.append(ys)
            atts.append(att)
        return torch.cat(preds, 1), torch.cat(atts, 1)


class LanguageCriterion(nn.Module):
    def __init__(self):
        super(LanguageCriterion, self).__init__()

    def forward(self, pred, target, mask):
        pred = pred.contiguous().view(-1, pred.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        # 交叉熵
        mask = mask.float()
        output = -pred.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


from seq2seq.data_loader import train_dataloader, en_total_words, cn_total_words, test_dataloader, decode_sents, BOS, \
    cn_dict

hidden_size = 100
embedding_size = 100
encoder = Encoder(en_total_words, embedding_size, hidden_size, hidden_size)
decoder = Decoder(cn_total_words, embedding_size, hidden_size, hidden_size)
model = Seq2Seq(encoder, decoder)
loss_fn = LanguageCriterion()
optimizer = torch.optim.Adam(model.parameters())


def eval():
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for i, (en, en_lens, cn, cn_lens) in enumerate(test_dataloader):
            cn_input = cn[:, :-1]  # 去掉最后一个字符
            cn_output = cn[:, 1:]  # 输出从第一个字符开始，不是<BOS>; <BOS> -> 第一个字符；最后一个字符-> <EOS>
            cn_lens = cn_lens - 1
            cn_lens[cn_lens <= 0] = 1  # 仅保留[<BOS>]

            pred, _ = model(en, en_lens, cn_input, cn_lens)
            # 只计算句子长度之内的损失。
            output_mask = torch.arange(cn_lens.max().item())[None, :] < cn_lens[:, None]
            loss = loss_fn(pred, cn_output, output_mask)

            num_words = torch.sum(cn_lens).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss / total_num_words)


def train(num_epoch=10):
    print(torch.__config__.parallel_info())
    for epoch in range(num_epoch):
        model.train()
        total_num_words = 0
        total_loss = 0
        for i, (en, en_lens, cn, cn_lens) in enumerate(train_dataloader):
            cn_input = cn[:, :-1]  # 去掉最后一个字符
            cn_output = cn[:, 1:]  # 输出从第一个字符开始，不是<BOS>; <BOS> -> 第一个字符；最后一个字符-> <EOS>
            cn_lens = cn_lens - 1
            cn_lens[cn_lens <= 0] = 1  # 修补长度为1的句子

            # cn_input: 全部传入，可做teacher，优化训练
            # 输出：[batch, sentence_len , pb_on_cn_words]
            pred, _ = model(en, en_lens, cn_input, cn_lens)
            # 只计算句子长度之内的损失。
            output_mask = torch.arange(cn_lens.max().item())[None, :] < cn_lens[:, None]
            loss = loss_fn(pred, cn_output, output_mask)

            num_words = torch.sum(cn_lens).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if i % 100 == 0:
                print("Epoch", epoch, "iteration", i, " training loss", loss.item())
                translate()
        eval()
        torch.save(model.state_dict(), f"rnn_attention_model_epoch_{epoch}.model")

    print("Epoch", epoch, "Training loss", total_loss / total_num_words)


def translate():
    # model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for i, (en, en_lens, cn, cn_lens) in enumerate(test_dataloader):
            # cn_input = cn[:, :-1]  # 去掉最后一个字符
            cn_output = cn[:, 1:]  # 输出从第一个字符开始，不是<BOS>; <BOS> -> 第一个字符；最后一个字符-> <EOS>
            # cn_lens = cn_lens - 1
            # cn_lens[cn_lens <= 0] = 1  # 仅保留[<BOS>]
            bos = torch.zeros(en.size(0), 1)
            bos.fill_(cn_dict[BOS])

            # cn_input: 全部传入，可做teacher，优化训练
            # 输出：[batch, sentence_len , pb_on_cn_words]
            pred, atts = model.translate(en, en_lens, cn_output)
            print(decode_sents(en, False))
            print(decode_sents(cn))
            print(decode_sents(pred))
            return


if __name__ == '__main__':
    # translate()
    train()
