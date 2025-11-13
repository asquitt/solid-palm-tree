"""Text preprocessing utilities."""

class TextPreprocessor:
    """Preprocessing for text data."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def preprocess(self, texts, max_length=512):
        """Tokenize and preprocess texts."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
