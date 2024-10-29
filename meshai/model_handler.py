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

class TextModelHandler(BaseModelHandler):
    def __init__(
        self,
        model_name_or_path='distilbert-base-uncased',
        num_labels=2,
        logger=None,
        model=None,
        tokenizer=None
    ):
        """
        Initializes the text model handler with a pre-trained or custom model.
        """
        super().__init__(logger)
        self.num_labels = num_labels

        # Detect available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.logger.info("Using MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU device.")

        if model and tokenizer:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.logger.info("Initialized with custom model and tokenizer.")
        else:
            self.model_name_or_path = model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels
            ).to(self.device)
            self.logger.info(f"Initialized with pre-trained model '{self.model_name_or_path}' on {self.device}.")

    def train(
        self,
        train_dataset,
        val_dataset=None,
        epochs=10,
        batch_size=16,
        output_dir='./text_model_output'
    ):
        """
        Trains the text model.
        """
        self.logger.info("Starting training...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            eval_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True if val_dataset else False,
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy='epoch',  # Removed to prevent duplication
        )

        # Initialize Trainer with EarlyStoppingCallback if validation is provided
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else []

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
            callbacks=callbacks,
            compute_metrics=self.compute_metrics if val_dataset else None,
        )

        trainer.train()
        self.logger.info("Training completed.")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # Convert logits to predictions
        predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()

        # Convert labels to NumPy if not already
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

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
