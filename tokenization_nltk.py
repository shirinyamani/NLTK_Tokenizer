# copyright 2023 by Shirin Yamani
# This code is based on NLTK library and the Transformer library
# the goal id to leverage the 'PreTrainedTokenizer' class from 
# the Transformers library to create a NLTK-based tokenizer.

from typing import Dict, List, Tuple, Optional
from transformers import PreTrainedTokenizer
import nltk
from nltk.tokenize import word_tokenize
import os

class NlktTokenizer(PreTrainedTokenizer):

    """ Explanation: 
    The NlktTokenizer class is a custom tokenizer that combines the robustness of 
    NLTK's word_tokenize method with the features of the HuggingFace 
    Transformers (PreTrainedTokenizer) library. Therefore the overal goal is to  
    Construct a word-level encoding NLTK Tokenizer.

    Tokenization Process:
    1. Word-Level Tokenization: Utilizes NLTK's word_tokenize to break text into 
    individual words and punctuation, preserving the fundamental linguistic units in 
    their original form. Example: "Hello, world!" -> ['Hello', ',', 'world', '!']

    2. Handling Special Tokens: Capable of recognizing and appropriately inserting
    special tokens (like <s> for start of sentence) that are essential for 
    Transformer-based models to understand textual context and sentence boundaries.
    Example: "<s> Hello, world! <end_of_text>" -> ['<s>', 'Hello', ',', 'world', '!', '<end_of_text>']

    3. Tokens and IDs Conversion: Facilitates the conversion between tokens and their
    unique IDs, aligning with the input requirements of Transformer models. The 
    tokenizer's vocabulary supports this conversion, enabling seamless model integration.
    Example: 'world' -> ID (based on vocabulary)

    4. Customizability and Vocabulary Management: Allows users to save and update the 
    tokenizer's vocabulary, providing adaptability for specific linguistic datasets or 
    specialized use cases. Example: Adding 'emoji' to the vocabulary.

    5. Preservation of Linguistic Integrity: Maintains the semantic and syntactic essence of 
    the text by accurately retaining original word forms and identifying punctuation, which is 
    critical for effective NLP applications.Example: "CAN'T" remains "CAN'T" and not split 
    into 'can' and 't'.

    This tokenizer is well-suited for a wide range of NLP tasks, offering a balance between the 
    linguistic depth of NLTK and the advanced capabilities of Transformer-based models.

    Args:
        vocab_file (str): 
            File path to the tokenizer's vocabulary.
        eos_token (str, optional): 
            Token to denote the end of a sentence. Defaults to "<s>".
        **kwargs: 
            Additional keyword arguments for the tokenizer.
    """
    
    def __init__(self, vocab_file='vocab.txt', eos_token="<s>", **kwargs):
        """
        Initialize the NlktTokenizer with a vocabulary file and an optional EOS token.
        """
        super().__init__(eos_token=eos_token, **kwargs)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = {line.strip(): idx for idx, line in enumerate(f.readlines())}
        self.id_to_token = {id: token for token, id in self.vocab.items()}

    def _tokenize(self, text, **kwargs):
        
        """
        Args:
            text (str): 
                The input text to tokenize.
            **kwargs: 
                Additional keyword arguments.
        Returns:
            List[str]: 
                List of tokens including special tokens.
        """
        
        nltk.download('punkt', quiet=True)
        special_tokens = [self.eos_token, "<end_of_text>"]
        temp_replacements = {}

        # Update to ensure special tokens are surrounded by spaces for proper separation
        for idx, token in enumerate(special_tokens):
            placeholder = f" __SPECIAL{idx}__ "
            if token in text:
                text = text.replace(token, placeholder)
                temp_replacements[placeholder.strip()] = token

        tokens = word_tokenize(text)
        reverted_tokens = [temp_replacements.get(token, token) for token in tokens]
        return reverted_tokens

    def tokenize(self, text, add_special_tokens=False, **kwargs) -> List[str]:
        
        """
        Tokenizes the input text and optionally adds special tokens in case there is a special 
        token. This method wraps the _tokenize method, adding functionality for special tokens.

        Args:
            text (str): 
                The text to tokenize.
            add_special_tokens (bool, optional): 
                Whether to add special tokens. Defaults to False.
            **kwargs: 
                Additional keyword arguments.
        Returns:
            List[str]: The list of tokens.
        """

        tokens = self._tokenize(text, **kwargs)
        if add_special_tokens:
            if self.eos_token not in tokens:
                tokens.insert(0, self.eos_token)
            if "<end_of_text>" not in tokens:
                tokens.append("<end_of_text>")
        return tokens
    
    def _convert_token_to_id(self, token):

        """
        Converts a token(str) to its corresponding ID reflected in vocab.txt file

        Args:
            token (str): 
                The token to convert.
        Returns:
            int: 
                Token ID, or ID of unknown token if not found.
        """
        return self.vocab.get(token, self.vocab.get("[UNK]")) + 1 if token in self.vocab else self.vocab.get("[UNK]")

    def _convert_id_to_token(self, index: int) -> str:
    
        """
        Converts an ID (int) to its corresponding token (str) using the vocab.

        Args:
            index (int): 
                Token ID to convert.
        Returns:
            str: 
                Corresponding token, or unknown token if ID not found.
        """
        adjusted_index = index - 1  # Adjust for zero-based indexing
        return self.id_to_token.get(adjusted_index, "[UNK]")

    def vocab_size(self):
    
        """
        Returns the size of the tokenizer's vocabulary.
        """
        return len(self.vocab)
    
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:

        """
        Save the tokenizer's vocabulary to a specified directory.

        Args:
            save_directory (`str`): 
                The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str], optional): 
                A prefix for the filename. Defaults to None.
        Returns:
            `Tuple(str)`: 
                The path to the saved vocabulary file.
        """
        if not filename_prefix:
            filename_prefix = ""
        vocab_file = os.path.join(save_directory, filename_prefix + "vocab.txt")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in self.vocab.keys():
                f.write(token + '\n')
        return (vocab_file,)
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Returns a dictionary of tokens and their corresponding IDs, adjusted for zero-based indexing.
        """
        vocab = {self._convert_id_to_token(i + 1): i for i in range(self.vocab_size())}

        # If the tokenizer has any added tokens, update the vocab dictionary with these
        if hasattr(self, 'added_tokens_encoder'):
            vocab.update({token: i + len(vocab) for token, i in self.added_tokens_encoder.items()})

        return vocab
