import re
from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(
        self,
        text: str
    ) -> List[str]:
        pass


class CharTokenizer(Tokenizer):

    def tokenize(self, text: str) -> List[str]:
        return list(text)
    

class WordTokenizer(Tokenizer):

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for word in text.split():
            tokens.extend(self._split_punctuation(word))
        return tokens

    def _split_punctuation(self, token: str) -> List[str]:
        return re.findall(r"(\w+|[^\w\s])", token)
    

class SubwordTokenizer(Tokenizer):

    def tokenize(self, text: str) -> List[str]:
        return ["_" + w for w in text.split()]