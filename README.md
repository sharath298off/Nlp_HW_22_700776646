# Bigram Language Model Implementation

**Name:** Sharath Chandra S
**Student ID:** 700776646

------------------------------------------------------------------------

## Program Overview

This program implements a Bigram Language Model using Maximum Likelihood
Estimation (MLE).\
It is based on a small training corpus containing three sentences:

1.  `<s>`{=html} I love NLP `</s>`{=html}\
2.  `<s>`{=html} I love deep learning `</s>`{=html}\
3.  `<s>`{=html} deep learning is fun `</s>`{=html}

The goal of the program is to: - Compute unigram counts - Compute bigram
counts - Estimate bigram probabilities using MLE - Calculate the
probability of a given sentence - Compare two sentences and determine
which one the model prefers

------------------------------------------------------------------------

## How the Program Works

### 1. Training Phase

The model reads the training corpus and:

-   Counts each word occurrence (Unigram Count)
-   Counts each pair of consecutive words (Bigram Count)

Bigram probability is computed using:

P(w2 \| w1) = Count(w1, w2) / Count(w1)

This is the Maximum Likelihood Estimation formula.

------------------------------------------------------------------------

### 2. Sentence Probability Calculation

To compute the probability of a sentence, the model:

-   Breaks the sentence into bigrams
-   Multiplies all corresponding bigram probabilities
-   Returns 0 if a bigram was not seen during training

Sentence probability formula:

P(sentence) = Π P(w_i \| w\_{i-1})

------------------------------------------------------------------------

### 3. Testing

The program tests the following two sentences:

-   `<s>`{=html} I love NLP `</s>`{=html}
-   `<s>`{=html} I love deep learning `</s>`{=html}

It calculates both probabilities and prints which sentence has the
higher probability.

If both probabilities are equal, the program states that the model has
no preference.

------------------------------------------------------------------------

## How to Run

Use the following command:

python bigram_language_model.py

------------------------------------------------------------------------

## Key Concepts Used

-   Natural Language Processing (NLP)
-   Unigram Model
-   Bigram Model
-   Maximum Likelihood Estimation (MLE)
-   Probability Computation

------------------------------------------------------------------------

## Conclusion

This implementation demonstrates how a simple statistical language model
works using bigram probabilities.\
It shows how sentence likelihood is determined purely based on observed
word transitions in a training corpus.
