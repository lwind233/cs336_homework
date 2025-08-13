import json
import regex as re


import json
import regex as re


class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.vocab_len = len(vocab)
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}

        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):

        with open(vocab_filepath,'r') as f:
                raw_vocab = json.load(f)

        vocab = {}
        for key,value in raw_vocab.items():
            vocab[value] = key.encode('utf-8')

        with open(merges_filepath,'r') as mf:
            raw_merges = mf.readlines()
        merges = []
        for i in raw_merges:
            content = i.split('\n')[0].split(' ')
            merges.append((content[0].encode('utf-8'),content[0].encode('utf-8')))

        return cls(vocab,merges,special_tokens)

    def encode(self,text): 
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            special_token = "|".join([re.escape(special_token) for special_token in self.special_tokens])
            raw_tokens = re.split(f'({special_token})',text)
        else:
            raw_tokens = [text]
        PATTERN = re.compile(PAT)

        result = []
        for text in raw_tokens:
            if text in self.special_tokens:
                result.append(self.vocab_reversed.get(text.encode('utf-8')))
                continue
            tokens = [match.group(0) for match in PATTERN.finditer(text)]
            for i in tokens:
                ei = i.encode('utf-8')
                b_token = [bytes([b]) for b in ei]
                if len(b_token) == 1 or (self.special_tokens and i in self.special_tokens):
                    # 当前字节串是特殊标记或者仅含有一个字节，此时直接进行遍历不会浪费太多时间
                    for iidx in range(self.vocab_len):
                        if self.vocab[iidx] == b_token[0]:
                            result.append(iidx)
                            break
                else:
                    # 参考https://blog.csdn.net/Bug_makerACE/article/details/149278120?spm=1001.2014.3001.5502
                    for pair in self.merges:
                        a, b = pair
                        ptr = 0
                        tmp_tok = a+b
                        new_tok = []
                        while(ptr<len(b_token)):
                            if ptr<len(b_token)-1 and (b_token[ptr],b_token[ptr+1]) == pair:
                                new_tok.append(tmp_tok)
                                ptr = ptr+2
                            else:
                                new_tok.append(b_token[ptr])
                                ptr = ptr+1
                        b_token = new_tok
                    for j in b_token:
                        result.append(self.vocab_reversed.get(j))

        return result

    def encode_iterable(self,iterable):
        for i in iterable:
            yield from self.encode(i)

    def decode(self,ids):
        token = bytes()
        for i in ids:
            token = token + self.vocab[i]
        return token.decode('utf-8',errors='replace')