
# NLTK Tokenizer for Transformers 🤗 

## 📖 Overview 
The NLTK Tokenizer is a custom tokenizer class designed for use with the Hugging Face Transformers library. This tokenizer leverage the `NlktTokenizer` class extends the `PreTrainedTokenizer` from the Hugging Face's Transformers library to create a [NLTK](https://www.nltk.org/index.html)-based tokenizer. This approach combines the robust pre-training and easy integration features of the `PreTrainedTokenizer` with the linguistic processing strengths of NLTK's `word_tokenize`. The result is a tokenizer that is both powerful in handling diverse language patterns and compatible with advanced NLP modeling techniques.


## 🛠️ Installation
To use the NLTK Tokenizer, ensure you have both `transformers` and `nltk` libraries installed. You can install them using:

- ### With pip
```bash
pip install transformers nltk
```
- ### With Conda 
```bash
conda install -c huggingface transformers nltk
```

## 🚴‍♂️ Getting Started 

### Initializing the Tokenizer
- Clone this repo
- Go to the directory where you cloned this repo
- Initialize the NLTK Tokenizer with a vocabulary file.  Note that your vocab file should list one token per lines:
```python
from tokenization_nltk import NlktTokenizer

tokenizer = NlktTokenizer(vocab_file='path/to/your/vocabulary.txt') #vocab.txt
```
- Enjoy 🤗

## 🔬 Basic Usage Examples

1. **Simple Tokenization:**

   ```python
   text = "Hello Shirin,  How are you?"
   tokens = tokenizer.tokenize(text)
   print("Tokens:", tokens) #ouput: Tokens: ['Hello', 'Shirin', ',', 'How', 'are', 'you', '?']
   ```

2. **Including Special Tokens:** 

   ```python
   text = "<s>Hello, world!<end_of_text>"
   tokens = tokenizer.tokenize(text, add_special_tokens=True)
   print(tokens) #output: ['<s>', 'Hello', ',', 'world', '!', '<end_of_text>']
   ```
   
3. **Token-ID Conversion:**

   ```python
   tokens = ['the', 'weather', 'IS', 'Sunny', '!']
   token_ids = [tokenizer.convert_tokens_to_ids(token.lower()) for token in tokens]#lower() because the vocab.txt is all in lower case for us
   print(token_ids) #output: [1997, 4634, 2004, 11560, 1000]
   ```

4. **ID-Token Conversion:**

   ```python
   ids = [1, 24707, 4634, 19238, 1000, 31000]
   tokens = [tokenizer.convert_ids_to_tokens(id) for id in ids]
   print(tokens) #output: ['[PAD]', 'cloudy', 'weather', 'sucks', '!', '[UNK]']
   ```

5. **Tokenizing a Long Text:**

    ```python
   long_text = "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."
   long_tokens = tokenizer.tokenize(long_text)
   print("Tokens:", long_tokens) #output: ['In', 'a', 'hole', 'in', 'the', 'ground', 'there', 'lived', 'a', 'hobbit', '.', 'Not', 'a', 'nasty', ',', 'dirty', ',', 'wet', 'hole', ',', 'filled', 'with', 'the', 'ends', 'of', 'worms', 'and', 'an', 'oozy', 'smell', ',', 'nor', 'yet', 'a', 'dry', ',', 'bare', ',', 'sandy', 'hole', 'with', 'nothing', 'in', 'it', 'to', 'sit', 'down', 'on', 'or', 'to', 'eat', ':', 'it', 'was', 'a', 'hobbit-hole', ',', 'and', 'that', 'means', 'comfort', '.']
   ```

6. **Tokenizing Sentences with Emojis:**

   ```python
   text_with_emoji = "I love pizza 🍕! Do you like it too?"
   tokens_with_emoji = tokenizer.tokenize(text_with_emoji)
   print("Tokens:", tokens_with_emoji) #output: ['I', 'love', 'pizza', '🍕', '!', 'Do', 'you', 'like', 'it', 'too', '?']
   ```

7. **Saving the Tokenizer:**

Save the tokenizer's state, including its vocabulary:

   ```python
   tokenizer.save_vocabulary(save_directory='path/to/save')
   ```

## 🧪 Evaluation using ```Pytest``` 
We have comprehensively tested our tokenizer by implementing various test cases using pytest, ensuring its robustness and functionality across different input scenarios. Make sure to try it yourself by:

   ```python
   pytest test_tokenization_nltk.py
   ```

## ⚠️ Limitations:
- **Contextual understanding:** Biggest concern with NLTK's tokenization is that it operates mainly at the word level! This means it does not capture nuanced tokenization decisions needed for some NLP tasks that require sub-word or character-level understanding!
- **Language Complexity:** NLTK might struggle with tokenizing languages with complex morphologies or those requiring specialized tokenization rules. For instance, it might struggle to handle languages that heavily rely on context, like some forms of Chinese or Japanese. 
- **Out-of-Vocabulary Words:** If the tokenizer encounters words not present in its vocabulary (like 31000 id in the last example),  it might use an [UNK] (unknown) token or handle them poorly, affecting downstream tasks' performance.
- **Limited Preprocessing Performance:** It does not support fully the emojis.


## 🤗 Hub Integration
Make sure you have your vocabularary file (```vocab.txt```) in the same directory where you have the project. 

1. **Simple Tokenization:**
```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ShirinYamani/task", trust_remote_code = True)
text = "Example sentence for tokenization."
# Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
```
2. **Including Special Tokens:** 

   ```python
   text = "<s>Hello, world!<end_of_text>"
   tokens = tokenizer.tokenize(text, add_special_tokens=True)
   print(tokens) #output: ['<s>', 'Hello', ',', 'world', '!', '<end_of_text>']
   ```
   
3. **Token-ID Conversion:**

   ```python
   tokens = ['the', 'weather', 'IS', 'Sunny', '!']
   token_ids = [tokenizer.convert_tokens_to_ids(token.lower()) for token in tokens]#lower() because the vocab.txt is all in lower case for us
   print(token_ids) #output: [1997, 4634, 2004, 11560, 1000]
   ```