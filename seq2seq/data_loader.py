from collections import Counter
import torch
import nltk

# if no punkt
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.download('punkt')

train_file = "../data/translate_train.txt"
dev_file = "../data/translate_dev.txt"
test_file = "../data/translate_test.txt"

BOS = "BOS"  # begin of sentence
EOS = "EOS"  # end of sentence
UNK = "UNK"
PAD = "PAD"

PAD_IDX = 0
UNK_IDX = 1


# 把句子切成单词数组,并且标记头尾
def load_data(in_file):
    cn = []
    en = []
    num_examples = 0
    with open(in_file, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            en.append(["BOS"] + nltk.word_tokenize(parts[0].lower()) + ["EOS"])
            # 没有分词
            cn.append(["BOS"] + list(parts[1]) + ["EOS"])
    return en, cn


train_en, train_cn = load_data(train_file)
dev_en, dev_cn = load_data(dev_file)


# 构建词典
def build_dict(sentences, max_words=500):
    counter = Counter()
    for sentence in sentences:
        for word in sentence:
            counter[word] += 1
    topn = counter.most_common(max_words)
    total_words = len(topn) + 2
    word_dict = {word[0]: i + 2 for i, word in enumerate(topn)}
    word_dict[PAD_IDX] = PAD
    word_dict[UNK_IDX] = UNK
    return word_dict, total_words


# word -> index
en_dict, en_total_words = build_dict(train_en, 5000)
cn_dict, cn_total_words = build_dict(train_cn, 5000)

print(f"en vocabulary size:{en_total_words}")
print(f"cn vocabulary size:{cn_total_words}")

# index -> index
en_dict_rev = {v: k for k, v in en_dict.items()}
cn_dict_rev = {v: k for k, v in cn_dict.items()}


def encode_sentences(sents, word_dict: dict):
    return [[word_dict.get(w, UNK_IDX) for w in s] for s in sents]


def decode_sentences(sents, word_dict_rev: dict):
    sents = sents.numpy()
    return [[word_dict_rev.get(w , UNK) for w in s] for s in sents]


def sort_sentences(en_sents, cn_sents):
    idx = sorted(range(len(en_sents)), key=lambda x: len(en_sents[x]))
    return [en_sents[i] for i in idx], [cn_sents[i] for i in idx]


class LanguageLoader:

    def __init__(self, file: str, batch_size=40, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_en, self.train_cn = load_data(file)
        self.sents_en = encode_sentences(self.train_en, en_dict)
        self.sents_cn = encode_sentences(self.train_cn, cn_dict)
        self.sents_en_lens = [len(v) for v in self.sents_en]
        self.sents_cn_lens = [len(v) for v in self.sents_cn]
        self.sents_en_lens_max = max(self.sents_en_lens)
        self.sents_cn_lens_max = max(self.sents_cn_lens)
        self._batch_index = 0
        self.batch_count = len(self.sents_en) // self.batch_size

    # 按最长的句子补齐短句子
    def pad_sentences(self, sentences):
        lens = torch.LongTensor([len(s) for s in sentences])
        max_len = torch.max(lens)
        result = torch.zeros([lens.size(0), max_len], dtype=torch.long)
        for i, sentence in enumerate(sentences):
            result[i, :lens[i]] = torch.IntTensor(sentence)
        return result, lens

    def get_batch(self, i: int):
        s = i * self.batch_size
        e = (i + 1) * self.batch_size
        x_batch, x_lens = self.pad_sentences(self.sents_en[s:e])
        y_batch, y_lens = self.pad_sentences(self.sents_cn[s:e])
        return x_batch, x_lens, y_batch, y_lens

    def __len__(self):
        return self.batch_count

    def __next__(self):
        self._batch_index = self._batch_index % len(self)
        r = self.get_batch(self._batch_index)
        self._batch_index += 1
        return r

    def __iter__(self):
        self._batch_index = 0
        return self


train_dataloader = LanguageLoader(train_file, batch_size=20)
test_dataloader = LanguageLoader(test_file, batch_size=20)



def decode_sents(sentences, is_cn=True):
    word_dict_rev = cn_dict_rev if is_cn else en_dict_rev
    r = decode_sentences(sentences, word_dict_rev=word_dict_rev)
    decoded_sents = []
    for v in r:
        sent = []
        for x in v :
            if x == EOS:
                break
            if x in [BOS , PAD]:
                continue
            sent.append(x)
        if is_cn:
            decoded_sents.append("".join(sent))
        else:
            decoded_sents.append(" ".join(sent))
    return decoded_sents



if __name__ == '__main__':
    for en, en_lens, cn, cn_lens in train_dataloader:
        print(decode_sentences(en, word_dict_rev=en_dict_rev))
        print(decode_sentences(cn, word_dict_rev=cn_dict_rev))
        exit(0)