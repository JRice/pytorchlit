from itertools import islice # Just for inspecting parts of a dictionary, since they don't have order and can't be sliced like lists
from matplotlib import pyplot as plt

import torch # We need tensors and probability distributions, so we'll use PyTorch for that

import torch.nn.functional as F # For the cross-entropy loss function, which we'll use to evaluate our model's performance
# It bugs me that we call it "F", but that's convention. Math nerds hate readable names, apparently.

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
BigramProbs = (N+1).float() / N.sum(dim=1, keepdim=True) # Convert counts to probabilities

def sample_bigram(N, string_to_int, int_to_string):
    index = 0
    name = ''

    while True:
        prob = BigramProbs[index]
        index = torch.multinomial(prob, num_samples=1, replacement=True, generator=generator).item()
        name += int_to_string[index]
        if index == 0:
            break
    print(f"Generated name: '{name}'")

print("### Simple Sample of Bigrams ###")

for _ in range(10):
    sample_bigram(N, string_to_int, int_to_string)

def evaluate_words(words_list, N, BigramProbs, string_to_int):
    negative_log_likelihood = 0.0
    count = 0
    for w in words_list:
        chars = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            int_ch1 = string_to_int[ch1]
            int_ch2 = string_to_int[ch2]
            prob = BigramProbs[int_ch1, int_ch2]
            log_prob = torch.log(prob)
            negative_log_likelihood -= log_prob
            count += 1
            # print(f"Bigram: '{ch1}{ch2}', Count: {N[int_ch1, int_ch2].item()}, Probability: {prob.item():.4f}, Log-Probability: {log_prob:.4f}")
    print(f"For the names {', '.join(words_list)}:")
    print(f"{negative_log_likelihood=:.4f}")
    print(f"Average negative log-likelihood: {negative_log_likelihood / count:.4f}")
    
evaluate_words(words[:3], N, BigramProbs, string_to_int)
evaluate_words(['grok'], N, BigramProbs, string_to_int)
evaluate_words(['jeremyq'], N, BigramProbs, string_to_int)

def evaluate_neuron(word, xs, ys, prob):
    nlls = torch.zeros(len(word))
    for i in range(len(word)):
        x = xs[i].item()
        y = ys[i].item()
        print("-----")
        print(f"Bigram example {i+1}: '{int_to_string[x]}{int_to_string[y]}'")
        print(f"Input: '{int_to_string[x]}' ({x})")
        print('Output probabilities from the NN:', prob[i])
        print(f"label: '{int_to_string[y]}' ({y})")
        p = prob[i, y]
        print(f"Probability of correct next character '{int_to_string[y]}': {p:.4f}")
        logp = torch.log(p)
        print(f"Log-probability of correct next character '{int_to_string[y]}': {logp:.4f}")
        nll = -logp
        print(f"Negative log-likelihood: {nll:.4f}")
        nlls[i] = nll
    print(f"{nlls.sum()=: .4f}")
    print(f"Average negative log-likelihood: {nlls.mean().item():.4f}")


def forward_pass(xs, W, num):
    xenc = F.one_hot(xs, num_classes=len(string_to_int)).float()
    logits = xenc @ W # Matrix multiplication to get the logits for the next character
    counts = logits.exp() # Convert logits to counts by taking the exponential
    probs = counts / counts.sum(1, keepdim=True) # Convert counts to probabilities
    loss = -probs[torch.arange(num), ys].log().mean()
    print(f"{loss.item()=: .4f}")
    return loss

def backward_pass(W, loss):
    W.grad = None
    loss.backward()

def update(W, lr):
    W.data += -lr * W.grad # Update the weights using gradient descent

print('### Neural Network Approach ###')

# Conventionally, x for input, y for output:
xs, ys = [], []
for w in words:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        int_ch1 = string_to_int[ch1]
        int_ch2 = string_to_int[ch2]
        xs.append(int_ch1)
        ys.append(int_ch2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f"Number of examples: {num}")
W = torch.randn(len(string_to_int), len(string_to_int), generator=generator, requires_grad=True) # One-hot encode the input characters

for k in range(200):
    loss = forward_pass(xs, W, num)
    backward_pass(W, loss)
    update(W, 50)
# Final check:
loss = forward_pass(xs, W, num)

