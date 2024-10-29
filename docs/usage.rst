Usage
=====

### Initialize the TextModelHandler with a Configuration File

```python
from meshai.model_handler import TextModelHandler

# Initialize with configuration file
handler = TextModelHandler(config_path='config.json')

# Assume train_dataset and val_dataset are predefined
handler.train(train_dataset, val_dataset=val_dataset)

# Make predictions
predictions, probabilities = handler.predict(["Sample text for classification."])

