import pickle
from domain.vocabulary import Vocabulary


class PickleVocabularyRepository:
    def save(self, vocab: Vocabulary, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(vocab, f)

    def load(self, path: str) -> Vocabulary:
        with open(path, "rb") as f:
            return pickle.load(f)
