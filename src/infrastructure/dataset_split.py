import random
from typing import List, Tuple


def split_dataset(
        texts: List[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)

    n_train = int(len(texts) * train_ratio)
    n_val = int(len(texts) * val_ratio)

    train = [texts[i] for i in indices[:n_train]]
    val = [texts[i] for i in indices[n_train:n_train + n_val]]
    test = [texts[i] for i in indices[n_train + n_val:]]

    return train, val, test