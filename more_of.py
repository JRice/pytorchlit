from itertools import islice # Just for inspecting parts of a dictionary, since they don't have order and can't be sliced like lists
from matplotlib import pyplot as plt

import torch # We need tensors and probability distributions, so we'll use PyTorch for that

# Following along with Karpathy again: https://www.youtube.com/watch?v=PaCmpygFfXo

words = open('data/input.txt', 'r').read().splitlines()
print(f"First 10 words: {words[:10]}")
print(f"Number of words: {len(words)}")
print(f"Shortest word: {min(len(w) for w in words)}")
print(f"Longest word: {max(len(w) for w in words)}")

possible_chars = sorted(list(set(''.join(words))))
string_to_int = {s: i+1 for i, s in enumerate(possible_chars)}
string_to_int['.'] = 0 # Start or end of word

int_to_string = {i: s for s, i in string_to_int.items()}

N = torch.zeros((len(string_to_int), len(string_to_int)), dtype=torch.int32)

for w in words:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        int_ch1 = string_to_int[ch1]
        int_ch2 = string_to_int[ch2]
        N[int_ch1, int_ch2] += 1

# Just showing the 10 most common bigrams:
# print(dict(islice(sorted(bigrams.items(), key = lambda kv: -kv[1]), 10)))

# Simple heat-map. Neat, but hard to read without labels.
# plt.imshow(N)
# plt.show()

def show_heatmap(N, string_to_int, int_to_string):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')
    for i in range(len(string_to_int)):
        for j in range(len(string_to_int)):
            ch_str = int_to_string[i] + int_to_string[j]
            plt.text(j, i, ch_str, ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
    plt.axis('off')
    plt.show()

generator = torch.Generator().manual_seed(2147483647) # Just for reproducibility
# Note we add one to the counts to avoid zero probabilities, which would cause issues when we take the log of the probabilities later on.
Probs = (N+1).float() / N.sum(dim=1, keepdim=True) # Convert counts to probabilities

def sample_bigram(N, string_to_int, int_to_string):
    index = 0
    name = ''

    while True:
        prob = Probs[index]
        index = torch.multinomial(prob, num_samples=1, replacement=True, generator=generator).item()
        name += int_to_string[index]
        if index == 0:
            break
    print(f"Generated name: '{name}'")

for _ in range(10):
    sample_bigram(N, string_to_int, int_to_string)

def evaluate_words(words_list, N, Probs, string_to_int):
    negative_log_likelihood = 0.0
    count = 0
    for w in words_list:
        chars = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            int_ch1 = string_to_int[ch1]
            int_ch2 = string_to_int[ch2]
            prob = Probs[int_ch1, int_ch2]
            log_prob = torch.log(prob)
            negative_log_likelihood -= log_prob
            count += 1
            # print(f"Bigram: '{ch1}{ch2}', Count: {N[int_ch1, int_ch2].item()}, Probability: {prob.item():.4f}, Log-Probability: {log_prob:.4f}")
    print(f"For the names {', '.join(words_list)}:")
    print(f"{negative_log_likelihood=:.4f}")
    print(f"Average negative log-likelihood: {negative_log_likelihood / count:.4f}")
    
evaluate_words(words[:3], N, Probs, string_to_int)
evaluate_words(['grok'], N, Probs, string_to_int)
evaluate_words(['jeremyq'], N, Probs, string_to_int)
