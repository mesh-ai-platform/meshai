from meshai.data_handler import TextDataset
from meshai.model_handler import TextModelHandler
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Load data
    texts = ["I love this product!", "This is the worst experience ever."]
    labels = [1, 0]  # 1: Positive, 0: Negative

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Initialize the model handler
    model_handler = TextModelHandler(model_name_or_path='bert-base-uncased', num_labels=2)

    # Prepare datasets
    train_dataset = TextDataset(train_texts, train_labels, model_handler.tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, model_handler.tokenizer)

    # Train the model
    model_handler.train(train_dataset, val_dataset, epochs=2)

    # Save the model
    model_handler.save_model('./text_model')

    # Load the model
    model_handler.load_model('./text_model')

    # Make predictions
    test_texts = ["Absolutely fantastic service."]
    predictions, probabilities = model_handler.predict(test_texts)
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        sentiment = 'Positive' if pred.item() == 1 else 'Negative'
        confidence = prob.max().item()
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}\n")

if __name__ == '__main__':
    main()
