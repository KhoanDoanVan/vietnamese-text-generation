

from experiments.config import ExperimentConfig
import os
from typing import List, Tuple
import random

class PrepareData:

    """
    Prepare data pipeline
    """


    def prepare_data(self, config: ExperimentConfig):
        
        # Load corpus
        texts = self.load_vietnamese_corpus(config.data)

        # Split data
        train_texts, val_texts, test_texts = self.create_dataset_splits(
            texts,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

        
    

    def load_vietnamese_corpus(self, filepath: str, max_samples: int = None) -> list:
        
        if not os.path.exists(filepath):
            print(f"Warning: Corpus file not found at {filepath}")
            print("Creating sample Vietnamese corpus for demo...")

            # Sample Vietnamese sentences for demo
            sample_corpus = [
                "Việt Nam là một quốc gia nằm ở phía đông của bán đảo Đông Dương.",
                "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam.",
                "Tiếng Việt là ngôn ngữ chính thức của Việt Nam.",
                "Phở là món ăn truyền thống nổi tiếng của người Việt.",
                "Văn học Việt Nam có lịch sử phát triển lâu đời.",
                "Sông Mekong chảy qua miền Nam Việt Nam.",
                "Vịnh Hạ Long là di sản thiên nhiên thế giới.",
                "Cà phê Việt Nam nổi tiếng trên toàn thế giới.",
            ] * 100

            return sample_corpus
        
        with open(filepath, 'r', encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if max_samples:
            texts = texts[:max_samples]

        return texts
    

    def create_dataset_splits(
        self,
        texts: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[
        List[str],List[str],List[str]
    ]:
        """
        Dataset Spliting (train/val/test) 
        """

        random.seed(seed)

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        n = len(texts)
        indices = list(range(n))
        random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_texts = [
            texts[i] for i in indices[:train_end]
        ]
        val_texts = [
            texts[i] for i in indices[train_end:val_end]
        ]
        test_texts = [
            texts[i] for i in indices[val_end:]
        ]

        return train_texts, val_texts, test_texts