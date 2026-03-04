
# Bigram Language Model Implementation (MLE)
# Corpus: "I love NLP" activity dataset

from collections import defaultdict
from typing import List


class BigramLanguageModel:
    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.bigram_probabilities = {}

    def train(self, sentences: List[List[str]]) -> None:
        """Train the model by computing unigram and bigram counts."""
        for sentence in sentences:
            for i in range(len(sentence)):
                # Count unigram
                self.unigram_counts[sentence[i]] += 1

                # Count bigram (if not last word)
                if i < len(sentence) - 1:
                    bigram = (sentence[i], sentence[i + 1])
                    self.bigram_counts[bigram] += 1

        # Compute MLE bigram probabilities
        for (w1, w2), count in self.bigram_counts.items():
            self.bigram_probabilities[(w1, w2)] = (
                count / self.unigram_counts[w1]
            )

    def sentence_probability(self, sentence: List[str]) -> float:
        """Compute probability of a sentence using bigram MLE."""
        probability = 1.0

        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])

            if bigram not in self.bigram_probabilities:
                return 0.0

            probability *= self.bigram_probabilities[bigram]

        return probability


def main():
    # Training corpus
    corpus = [
        ["<s>", "I", "love", "NLP", "</s>"],
        ["<s>", "I", "love", "deep", "learning", "</s>"],
        ["<s>", "deep", "learning", "is", "fun", "</s>"]
    ]

    # Initialize and train model
    model = BigramLanguageModel()
    model.train(corpus)

    print("Unigram Counts:")
    for word, count in model.unigram_counts.items():
        print(f"{word}: {count}")

    print("\nBigram Counts:")
    for bigram, count in model.bigram_counts.items():
        print(f"{bigram}: {count}")

    print("\nBigram Probabilities (MLE):")
    for bigram, prob in model.bigram_probabilities.items():
        print(f"P({bigram[1]} | {bigram[0]}) = {prob:.6f}")

    # Test sentences
    sentence1 = ["<s>", "I", "love", "NLP", "</s>"]
    sentence2 = ["<s>", "I", "love", "deep", "learning", "</s>"]

    prob1 = model.sentence_probability(sentence1)
    prob2 = model.sentence_probability(sentence2)

    print("\nSentence Probabilities:")
    print(f"P(S1) = {prob1:.6f}")
    print(f"P(S2) = {prob2:.6f}")

    if prob1 > prob2:
        print("\nModel prefers Sentence 1 because it has higher probability.")
    elif prob2 > prob1:
        print("\nModel prefers Sentence 2 because it has higher probability.")
    else:
        print("\nBoth sentences have equal probability under this model.")


if __name__ == "__main__":
    main()
