# meshai/model_handler.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchvision import models
import torch.nn as nn
import joblib
import os

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
    def __init__(self, model_name_or_path='distilbert-base-uncased', num_labels=2, logger=None, model=None, tokenizer=None):
        """
        Initializes the text model handler with a pre-trained or custom model.
        """
        super().__init__(logger)
        self.num_labels = num_labels

        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
            self.logger.info("Initialized with custom model and tokenizer.")
        else:
            self.model_name_or_path = model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, num_labels=self.num_labels
            )
            self.logger.info(f"Initialized with pre-trained model '{self.model_name_or_path}'.")

    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=8, output_dir='./text_model_output'):
        """
        Trains the text model.
        """
        from transformers import Trainer, TrainingArguments

        self.logger.info("Starting training...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            evaluation_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True if val_dataset else False,
            save_total_limit=1
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
        )

        trainer.train()
        self.logger.info("Training completed.")

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
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.logger.info(f"Model loaded from {load_path}")

    def predict(self, texts):
        """
        Makes predictions on text data.
        """
        self.model.eval()
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        self.logger.info("Prediction made.")
        return predictions, probabilities

class ImageModelHandler(BaseModelHandler):
    def __init__(self, model_name='resnet18', num_classes=2, logger=None, model=None):
        """
        Initializes the image model handler with a pre-trained or custom model.
        """
        super().__init__(logger)
        self.num_classes = num_classes

        if model:
            self.model = model
            self.logger.info("Initialized with custom image model.")
        else:
            self.model_name = model_name
            self.model = getattr(models, self.model_name)(pretrained=True)
            # Modify the last layer to match the number of classes
            if hasattr(self.model, 'fc'):
                self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            elif hasattr(self.model, 'classifier'):
                self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.num_classes)
            else:
                raise ValueError("Unknown model architecture.")
            self.logger.info(f"Initialized with pre-trained model '{self.model_name}'.")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, val_loader=None, epochs=3):
        """
        Trains the image model.
        """
        self.logger.info("Starting image model training...")
        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
            if val_loader:
                self.evaluate(val_loader)
        self.logger.info("Image model training completed.")

    def evaluate(self, val_loader):
        """
        Evaluates the image model.
        """
        self.logger.info("Evaluating image model...")
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        self.logger.info(f'Validation Accuracy: {accuracy}%')
        print(f'Validation Accuracy: {accuracy}%')

    def save_model(self, save_path):
        """
        Saves the image model.
        """
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Loads the image model.
        """
        self.model.load_state_dict(torch.load(load_path))
        self.logger.info(f"Model loaded from {load_path}")

    def predict(self, images):
        """
        Makes predictions on image data.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        self.logger.info("Prediction made.")
        return predictions, probabilities

class NumericalModelHandler(BaseModelHandler):
    def __init__(self, model=None, logger=None):
        """
        Initializes the numerical model handler.
        """
        super().__init__(logger)
        from sklearn.ensemble import RandomForestClassifier  # Moved import here to avoid unnecessary dependency
        self.model = model or RandomForestClassifier()
        self.logger.info(f"Initialized with model: {self.model.__class__.__name__}")

    def train(self, X_train, y_train):
        """
        Trains the numerical model.
        """
        self.logger.info("Starting training of numerical model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Numerical model training completed.")

    def save_model(self, save_path):
        """
        Saves the numerical model.
        """
        joblib.dump(self.model, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Loads the numerical model.
        """
        self.model = joblib.load(load_path)
        self.logger.info(f"Model loaded from {load_path}")

    def predict(self, X):
        """
        Makes predictions on numerical data.
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        self.logger.info("Prediction made.")
        return predictions, probabilities
