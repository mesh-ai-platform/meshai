# examples/example_usage.py
from meshai.data_handler import TextDataset
from meshai.model_handler import TextModelHandler
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Sample text data
    data = {
        'text': [
            'I love this product!',
            'This is the worst experience ever.',
            'Absolutely fantastic service.',
            'Not happy with the quality.',
            'The item exceeded my expectations.',
            'Terrible customer support.',
            'I am extremely satisfied with the purchase.',
            'This is not what I ordered.',
            'Great value for money.',
            'I want a refund!'
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
    }
    df = pd.DataFrame(data)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Specify the pre-trained model
    model_name = 'bert-base-uncased'  # You can choose any model from Hugging Face model hub
    num_labels = 2  # Binary classification

    # Initialize the model handler
    model_handler = TextModelHandler(model_name=model_name, num_labels=num_labels)

    # Prepare datasets
    train_dataset = TextDataset(train_texts.tolist(), train_labels.tolist(), model_handler.tokenizer)
    val_dataset = TextDataset(val_texts.tolist(), val_labels.tolist(), model_handler.tokenizer)

    # Train the model
    model_handler.train(train_dataset, val_dataset, epochs=2)

    # Save the model
    model_handler.save_model('./text_model')

    # Load the model (optional)
    model_handler.load_model('./text_model')

    # Make predictions
    test_texts = [
        'The product is excellent and works like a charm.',
        'I am disappointed with the service.',
        'Best purchase I have made this year.',
        'The item broke after one use.'
    ]
    predictions, probabilities = model_handler.predict(test_texts)
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        sentiment = 'Positive' if pred.item() == 1 else 'Negative'
        confidence = prob.max().item()
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}\n")

if __name__ == '__main__':
    main()
