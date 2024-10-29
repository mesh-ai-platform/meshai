# meshai/model_handler.py

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torchvision import models
import torch.nn as nn
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path

from meshai.logger import setup_logger

class BaseModelHandler:
    def __init__(self, logger=None):
        self.logger = logger or setup_logger()
        self.logger.info(f"Initialized {self.__class__.__name__}")

    def save_model(self, save_path):
        """
        Saves the model to the specified path.
        """
        raise NotImplementedError

    def load_model(self, load_path):
        """
        Loads the model from the specified path.
        """
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """
        Trains the model.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """
        Makes predictions using the model.
        """
        raise NotImplementedError

# meshai/model_handler.py (continued)

class TextModelHandler(BaseModelHandler):
    def __init__(self, config_path, logger=None):
        """
        Initializes the text model handler with configurations from a file.
        """
        super().__init__(logger)
        self.config = self.load_config(config_path)

        # Load model configuration
        self.model_name_or_path = self.config['model'].get('model_name_or_path', 'distilbert-base-uncased')
        self.num_labels = self.config['model'].get('num_labels', 2)

        # Load training configuration
        self.training_config = self.config.get('training', {})
        self.output_dir = self.training_config.get('output_dir', './text_model_output')

        # Device selection based on configuration or auto-detection
        self.device = self.select_device(self.config.get('device_preference', 'auto'))
        self.logger.info(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self.model = self.initialize_model()
        self.tokenizer = self.initialize_tokenizer()

    def load_config(self, config_path):
        """
        Loads configuration from a JSON or YAML file.
        """
        config_file = Path(config_path)
        if not config_file.is_file():
            self.logger.error(f"Configuration file {config_path} not found.")
            raise FileNotFoundError(f"Configuration file {config_path} not found.")

        try:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            self.validate_config(config)
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise e

    def validate_config(self, config):
        """
        Validates the configuration against a predefined schema.
        """
        from jsonschema import validate, ValidationError

        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "model_name_or_path": {"type": "string"},
                        "num_labels": {"type": "integer"}
                    },
                    "required": ["model_name_or_path", "num_labels"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "epochs": {"type": "integer"},
                        "batch_size": {"type": "integer"},
                        "learning_rate": {"type": "number"},
                        "weight_decay": {"type": "number"},
                        "warmup_steps": {"type": "integer"},
                        "gradient_clip_val": {"type": "number"},
                        "output_dir": {"type": "string"},
                        "logging_dir": {"type": "string"},
                        "save_total_limit": {"type": "integer"},
                        "early_stopping_patience": {"type": "integer"},
                        "logging_steps": {"type": "integer"},
                        "evaluation_strategy": {"type": "string"},
                        "save_strategy": {"type": "string"},
                        "load_best_model_at_end": {"type": "boolean"},
                        "evaluation_metrics": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["epochs", "batch_size", "learning_rate"]
                },
                "device_preference": {"type": "string"}
            },
            "required": ["model", "training"]
        }

        try:
            validate(instance=config, schema=schema)
            self.logger.info("Configuration validation passed.")
        except ValidationError as ve:
            self.logger.error(f"Configuration validation error: {ve.message}")
            raise ve

    def select_device(self, preference):
        """
        Selects the device based on user preference or auto-detection.
        """
        if preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preference == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preference == "cpu":
            return torch.device("cpu")
        else:
            if torch.backends.mps.is_available():
                self.logger.info("Auto-selecting MPS device.")
                return torch.device("mps")
            elif torch.cuda.is_available():
                self.logger.info("Auto-selecting CUDA device.")
                return torch.device("cuda")
            else:
                self.logger.info("Auto-selecting CPU device.")
                return torch.device("cpu")

    def initialize_model(self):
        """
        Initializes the model based on the configuration.
        """
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels
            ).to(self.device)
            self.logger.info(f"Initialized model '{self.model_name_or_path}' on {self.device}.")
            return model
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise e

    def initialize_tokenizer(self):
        """
        Initializes the tokenizer based on the configuration.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.logger.info(f"Initialized tokenizer for '{self.model_name_or_path}'.")
            return tokenizer
        except Exception as e:
            self.logger.error(f"Error initializing tokenizer: {e}")
            raise e

    def train(
        self,
        train_dataset,
        val_dataset=None
    ):
        """
        Trains the text model using configurations from the config file.
        """
        self.logger.info("Starting training...")

        # Extract training parameters
        training_args = TrainingArguments(
            output_dir=self.training_config.get('output_dir', './text_model_output'),
            num_train_epochs=self.training_config.get('epochs', 10),
            per_device_train_batch_size=self.training_config.get('batch_size', 16),
            eval_strategy=self.training_config.get('evaluation_strategy', 'epoch') if val_dataset else 'no',
            save_strategy=self.training_config.get('save_strategy', 'epoch'),
            logging_dir=self.training_config.get('logging_dir', './logs'),
            logging_steps=self.training_config.get('logging_steps', 10),
            load_best_model_at_end=self.training_config.get('load_best_model_at_end', False) if val_dataset else False,
            save_total_limit=self.training_config.get('save_total_limit', 2),
            learning_rate=self.training_config.get('learning_rate', 2e-5),
            weight_decay=self.training_config.get('weight_decay', 0.01),
            warmup_steps=self.training_config.get('warmup_steps', 500),
            gradient_clip_val=self.training_config.get('gradient_clip_val', 1.0),
        )

        # Compute class weights if class imbalance exists
        if val_dataset:
            from sklearn.utils.class_weight import compute_class_weight
            all_labels = [item['labels'] for item in train_dataset]
            all_labels = [label for sublist in all_labels for label in sublist]
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            self.logger.info(f"Computed class weights: {class_weights}")

            # Subclass Trainer to include class weights in loss
            class WeightedTrainer(Trainer):
                def __init__(self, weight, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.weight = weight

                def compute_loss(self, model, inputs, return_outputs=False):
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=self.weight)
                    loss = loss_fct(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.training_config.get('early_stopping_patience', 2))],
                compute_metrics=self.compute_metrics if 'evaluation_metrics' in self.training_config else None,
                weight=class_weights,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                callbacks=[],
                compute_metrics=self.compute_metrics if 'evaluation_metrics' in self.training_config else None,
            )

        trainer.train()
        self.logger.info("Training completed.")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
        labels = labels.numpy()

        metrics = {}
        available_metrics = {
            'accuracy': accuracy_score,
            'precision': lambda l, p: precision_recall_fscore_support(l, p, average='binary')[0],
            'recall': lambda l, p: precision_recall_fscore_support(l, p, average='binary')[1],
            'f1': lambda l, p: precision_recall_fscore_support(l, p, average='binary')[2],
        }

        for metric in self.training_config.get('evaluation_metrics', []):
            if metric in available_metrics:
                metrics[metric] = available_metrics[metric](labels, predictions)
            else:
                self.logger.warning(f"Unsupported metric '{metric}' requested.")

        return metrics


    def save_model(self, save_path):
        """
        Saves the text model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Loads the text model and tokenizer.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.logger.info(f"Model loaded from {load_path} on {self.device}.")

    def predict(self, texts):
        """
        Makes predictions on text data.
        """
        self.model.eval()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

        # Move inputs to the same device as the model
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        self.logger.info("Prediction made.")
        return predictions.cpu(), probabilities.cpu()
