from meshai.model_handler import TextModelHandler

def test_model_responses():
    # Initialize model handler (ensure the model is already trained and saved)
    model_handler = TextModelHandler(model_name_or_path='bert-base-uncased', num_labels=2)

    # Load the saved model (without decryption)
    model_handler.load_model('./text_model')

    # Define test cases
    test_texts = [
        "Absolutely fantastic service.",  # Positive
        "I am not happy with this product.",  # Negative
        "The experience was okay, not too bad.",  # Neutral/Uncertain
        "Worst service I have ever encountered.",  # Negative
        "Amazing product! I love using it every day.",  # Positive
    ]

    # Run predictions
    for text in test_texts:
        predictions, probabilities = model_handler.predict([text])
        sentiment = 'Positive' if predictions[0].item() == 1 else 'Negative'
        confidence = probabilities[0].max().item()

        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}\n")

if __name__ == '__main__':
    test_model_responses()
