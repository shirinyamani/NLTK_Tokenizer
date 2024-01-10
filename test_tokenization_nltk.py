import unittest
from .tokenization_nltk import NlktTokenizer

class NltkTokenizationTests(unittest.TestCase):
    """
    Test suite for the NlktTokenizer.

    This class contains various test cases to ensure the correct functionality of the NlktTokenizer.
    """

    def setUp(self):
        """
        Set up method to initialize the tokenizer before each test.

        It creates an instance of NlktTokenizer using a predefined vocabulary file.
        """
        super().setUp()
        vocab_path = 'vocab.txt'
        self.tokenizer = NlktTokenizer(vocab_file=vocab_path, trust_remote_code=True)
    
    def test_tokenization(self):
        """
        Test the basic tokenization capability.

        Ensures that standard sentences are tokenized correctly.
        """
        tokens = self.tokenizer.tokenize("This is a test")
        expected_tokens = ['This', 'is', 'a', 'test']
        self.assertEqual(tokens, expected_tokens)

    def test_empty_string(self):
        """
        Test tokenization of an empty string.

        Verifies that the tokenizer returns an empty list when given an empty string.
        """
        tokens = self.tokenizer.tokenize("")
        self.assertEqual(tokens, [])

    def test_special_characters(self):
        """
        Test tokenization of a string with special characters.

        Ensures that punctuation and special characters are handled correctly.
        """
        tokens = self.tokenizer.tokenize("Hello, world! #tokenizer")
        self.assertEqual(tokens, ["Hello", ",", "world", "!", "#", "tokenizer"])

    def test_numbers(self):
        """
        Test tokenization of a string containing numbers.

        Verifies that numbers are tokenized correctly, either as separate tokens or as part of text tokens.
        """
        tokens = self.tokenizer.tokenize("Testing 123, is it working?")
        self.assertEqual(tokens, ["Testing", "123", ",", "is", "it", "working", "?"])

    def test_mixed_case(self):
        """
        Test tokenization of a string with mixed case.

        Ensures that mixed case words are handled correctly and the case is preserved.
        """
        tokens = self.tokenizer.tokenize("Tokenizer Test: Mixed CASE.")
        self.assertEqual(tokens, ["Tokenizer", "Test", ":", "Mixed", "CASE", "."])
    
    def test_non_english_characters(self):
        """
        Test tokenization of a string with non-English characters.

        Checks if the tokenizer can handle non-English scripts, like Cyrillic, correctly.
        """
        tokens = self.tokenizer.tokenize("Привет, как дела? Café.")
        self.assertEqual(tokens, ["Привет", ",", "как", "дела", "?", "Café", "."])

    def test_long_text(self):
        """
        Test tokenization of a longer text.

        Verifies that the tokenizer can handle longer sentences with multiple elements correctly.
        """
        text = "This is a longer text. It includes multiple sentences, different punctuation marks, and various other elements."
        expected_tokens = ["This", "is", "a", "longer", "text", ".", "It", "includes", "multiple", "sentences", ",", "different", "punctuation", "marks", ",", "and", "various", "other", "elements", "."]
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_add_special_tokens(self):
        """
        Test the addition and usage of special tokens.

        Ensures that the tokenizer can handle added special tokens and includes them correctly during tokenization.
        """
        special_token = "<end_of_text>"
        self.tokenizer.add_tokens([special_token])
        tokens = self.tokenizer.tokenize("This is a test " + special_token)
        self.assertIn(special_token, tokens)
        
   
    def test_convert_token_to_id(self):
        """
        Test the conversion from token to its corresponding ID.

        Verifies that the method correctly returns the ID for a given token,
        accounting for the intentional increment in the method.
        """
        token = "test"
        expected_id = self.tokenizer.vocab.get(token) + 1  # Adjust for the increment in the method
        token_id = self.tokenizer._convert_token_to_id(token)
        self.assertEqual(token_id, expected_id, f"ID for token '{token}' should be {expected_id}, got {token_id}")

        # Testing an unknown token
        unknown_token = "unknown_token"
        unknown_token_id = self.tokenizer._convert_token_to_id(unknown_token)
        expected_unknown_id = self.tokenizer.vocab.get("[UNK]")  # No increment for unknown token
        self.assertEqual(unknown_token_id, expected_unknown_id, f"ID for unknown token should be {expected_unknown_id}, got {unknown_token_id}")

    def test_convert_id_to_token(self):
        """
        Test the conversion from ID to its corresponding token.

        Verifies that the method correctly returns the token for a given ID.
        """
        token_id = 5  # Assuming '5' is a valid ID in your vocabulary
        expected_token = self.tokenizer.id_to_token.get(token_id - 1)  # Adjusting for zero-based indexing
        token = self.tokenizer._convert_id_to_token(token_id)
        self.assertEqual(token, expected_token, f"Token for ID {token_id} should be '{expected_token}', got '{token}'")

        # Testing an unknown ID
        unknown_id = 99999  # Assuming this ID does not exist in your vocabulary
        unknown_token = self.tokenizer._convert_id_to_token(unknown_id)
        self.assertEqual(unknown_token, "[UNK]", f"Token for unknown ID should be '[UNK]', got '{unknown_token}'")
