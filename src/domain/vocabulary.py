from collections import Counter
from typing import Dict, List


class Vocabulary:

    PAD = "<pad>" # padding for batch
    UNK = "<unk>" # token doesn't exist in vocab
    BOS = "<bos>" # begin of sentence (Text generation need to start generate)
    EOS = "<eos>" # end of sentence


    def __init__(self, max_size: int, min_freq: int):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_freq = Counter()
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}


    def build(
            self,
            tokenized_texts: List[List[str]]
    ) -> None:
        
        """
        token2idx: token -> numerical
        idx2token: numerical -> token
        """


        """
        [["tôi", "yêu", "việt", "nam"],["việt", "nam", "rất", "đẹp"]]
        ==> {"việt": 2,"nam": 2,"tôi": 1,"yêu": 1,"rất": 1,"đẹp": 1}
        """
        for tokens in tokenized_texts:
            self.token_freq.update(tokens)


        tokens = [
            t for t,f in self.token_freq.items()
            if f >= self.min_freq # remove rare tokens -> reduce noise, overfitting 
        ]

        """
        - Token Appear a lot -> small index
        - small index -> embedding learn more efficient
        """
        tokens.sort(key=lambda t: self.token_freq[t], reverse=True)

        # spent space for special tokens(4 special)
        tokens = tokens[: self.max_size - 4]

        special = [
            self.PAD,
            self.UNK,
            self.BOS,
            self.EOS
        ]


        """
        assign index for special tokens
        {"<pad>": 0,"<unk>": 1,"<bos>": 2,"<eos>": 3}
        """
        self.token2idx = {
            tok: i for i,tok in enumerate(special)
        }


        """
        assign index for normal tokens
        {"<pad>": 0,"<unk>": 1,"<bos>": 2,"<eos>": 3,"việt": 4,"nam": 5}
        """
        for i,tok in enumerate(tokens, start=len(special)):
            self.token2idx[tok] = i


        # Reverse mapping
        self.idx2token = {
            i: t for t,i in self.token2idx.items()
        }


    
    def encode(
            self,
            tokens: List[str],
            add_special: bool = True
    ) -> List[int]:
        if add_special:
            # ["<bos>", "tôi", "yêu", "việt", "nam", "<eos>"]
            tokens = [self.BOS] + tokens + [self.EOS]

        """
        if token exist in vocab -> get index
        else -> map to <unk>

        exp: ["<bos>", "tôi", "yêu", "abc", "<eos>"]
        -> [2, 10, 11, 1, 3] (1 = <unk>)
        """
        return [
            self.token2idx.get(t, self.token2idx[self.UNK]) for t in tokens
        ]
    



    def decode(
            self,
            indices: List[int],
            skip_special: bool = True
    ) -> List[str]:
        
        """
        map index -> token
        index hasn't seen before -> <unk>

        exp: [2, 10, 11, 3]
        -> ["<bos>", "tôi", "yêu", "<eos>"]
        """
        tokens = [
            self.idx2token.get(i, self.UNK) for i in indices
        ]
        
        # remove special tokens
        if skip_special:
            tokens = [
                t for t in tokens
                if t not in {self.PAD, self.BOS, self.EOS}
            ]

        return tokens
