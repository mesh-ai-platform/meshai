# meshai/model_handler.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchvision import models
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class BaseModelHandler:
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
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        """
        Initializes the text model handler with a pre-trained model.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=8, output_dir='./text_model_output'):
        """
        Trains the text model.
        """
        from transformers import Trainer, TrainingArguments

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

    def save_model(self, save_path):
        """
        Saves the text model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, load_path):
        """
        Loads the text model and tokenizer.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

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
        return predictions, probabilities

class ImageModelHandler(BaseModelHandler):
    def __init__(self, model_name='resnet18', num_classes=2):
        """
        Initializes the image model handler with a pre-trained model.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = getattr(models, self.model_name)(pretrained=True)
        # Modify the last layer to match the number of classes
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        elif hasattr(self.model, 'classifier'):
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, self.num_classes)
        else:
            raise ValueError("Unknown model architecture.")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, val_loader=None, epochs=3):
        """
        Trains the image model.
        """
        for epoch in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
            if val_loader:
                self.evaluate(val_loader)

    def evaluate(self, val_loader):
        """
        Evaluates the image model.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total}%')

    def save_model(self, save_path):
        """
        Saves the image model.
        """
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        """
        Loads the image model.
        """
        self.model.load_state_dict(torch.load(load_path))

    def predict(self, images):
        """
        Makes predictions on image data.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities

class NumericalModelHandler(BaseModelHandler):
    def __init__(self, model=None):
        """
        Initializes the numerical model handler.
        """
        self.model = model or RandomForestClassifier()

    def train(self, X_train, y_train):
        """
        Trains the numerical model.
        """
        self.model.fit(X_train, y_train)

    def save_model(self, save_path):
        """
        Saves the numerical model.
        """
        joblib.dump(self.model, save_path)

    def load_model(self, load_path):
        """
        Loads the numerical model.
        """
        self.model = joblib.load(load_path)

    def predict(self, X):
        """
        Makes predictions on numerical data.
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities
