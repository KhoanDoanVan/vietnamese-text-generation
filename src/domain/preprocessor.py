from typing import List
from domain.normalizer import TextNormalizer
from domain.tokenizer import Tokenizer
from domain.vocabulary import Vocabulary



class TextPreprocessor:

    def __init__(
            self,
            normalizer: TextNormalizer,
            tokenizer: Tokenizer,
            vocabulary: Vocabulary
    ):
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.vocab = vocabulary


    def fit(self, texts: List[str]) -> None:
        tokenized = []
        for text in texts:
            norm = self.normalizer.normalize(text)
            tokenized.append(self.tokenizer.tokenize(norm))


    def encode(self, text: str) -> List[int]:
        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text)
        return self.vocab.encode(tokens)
    

    def decode(self,  indices: List[int]) -> str:
        tokens = self.vocab.decode(indices)
        return " ".join(tokens)