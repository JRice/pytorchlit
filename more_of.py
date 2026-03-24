from itertools import islice # Just for inspecting parts of a dictionary, since they don't have order and can't be sliced like lists
from matplotlib import pyplot as plt

import torch # At the moment, we just want to store tensors...

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

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')
for i in range(len(string_to_int)):
    for j in range(len(string_to_int)):
        ch_str = int_to_string[i] + int_to_string[j]
        plt.text(j, i, ch_str, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.show()