from experiments.config import ExperimentConfig
from models.vanilla_rnn import VanillaRNN
from models.lstm import LSTM
import torch.optim as optim
from training.trainer import LanguageModelTrainer
import os
import torch


class TrainModel:

    def train_model(
            self, 
            model_type: str, 
            config: ExperimentConfig,
            preprocessing,
            train_loader,
            val_loader
    ):
        
        print("\n" + "=" * 60)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("=" * 60)

        # Initialize Model
        if model_type == "rnn":
            model = VanillaRNN(
                vocab_size=len(preprocessing.vocab),
                embedding_dim=config.data["embedding_dim"],
                hidden_dim=config.data["hidden_dim"],
                num_layers=config.data["num_layers"],
                dropout=config.data["dropout"],
                tie_weights=config.data["tie_weights"]
            )
        elif model_type == "lstm":
            model = LSTM(
                vocab_size=len(preprocessing.vocab),
                embedding_dim=config.data["embedding_dim"],
                hidden_dim=config.data["hidden_dim"],
                num_layers=config.data["num_layers"],
                dropout=config.data["dropout"],
                tie_weights=config.data["tie_weights"]
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {model_type.upper()}")
        print(f"Parameters: {total_params:,}")
        print(f"Device: {config.device}")

        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training["learning_rate"]
        )

        # Learning rate Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

        # INitialize Trainer
        trainer = LanguageModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=config.device,
            grad_clip=config.training["grad_clip"]
        )

        # Train
        trainer.train(
            num_epochs=config.training["num_epochs"],
            scheduler=scheduler
        )

        # Save checkpoint
        checkpoint_path = os.path.join(
            config.paths,
            f'{model_type}_best.pt'
        )
        os.makedirs(config.paths, exist_ok=True)
        torch.save(trainer.best_model_state, checkpoint_path)
        print(f"\nBest model saved to {checkpoint_path}")

        # Load best model
        model.load_state_dict(trainer.best_model_state['model_state_dict'])

        return model, trainer