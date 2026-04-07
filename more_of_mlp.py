from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch # We need tensors and probability distributions, so we'll use PyTorch for that
import random

# Read in the words:
words = open('data/input.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 # Start or end of word
itos = {i: s for s, i in stoi.items()}

block_size = 3 # How many characters do we take to predict the next one?
hidden_layer_size = 100 # Number of neurons in the hidden layer of our MLP. This is a hyperparameter that we can tune.
vector_size = 2 # The dimensionality of the character embeddings. This is another hyperparameter that we can tune.
# learning_rate = 10**-0.8
minibatch_size = 32

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in word + '.': # We also want to predict the end of the word, which we'll represent with a dot
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(f"  {''.join(itos[i] for i in context)} ---> {itos[ix]} ({ch})")
            context = context[1:] + [ix] # Shift the context to the left and add the new character
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

# Training, development, and test splits, with Karpathy's infamously unhelpful names:
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


# This resets the network (which I want global, so not in a method):
g = torch.Generator().manual_seed(2147483647) # For reproducibility
C = torch.randn((len(stoi), vector_size), generator=g) # Each character gets mapped to a vector_size-dimensional vector
W1 = torch.randn([block_size * vector_size, hidden_layer_size], generator=g)
b1 = torch.randn(hidden_layer_size, generator=g)
W2 = torch.randn([hidden_layer_size, len(stoi)], generator=g)
b2 = torch.randn(len(stoi), generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
print(f"Number of parameters: {sum(p.numel() for p in parameters)}")

def forward(X, Y, ix=None):
    emb = C[X] if ix is None else C[X[ix]]
    h = torch.tanh(emb.view(-1, block_size * vector_size) @ W1 + b1) # The hidden layer activations, efficiently viewed.
    logits = h @ W2 + b2 # The output layer activations, which we'll interpret as unnormalized log probabilities for each character in the vocabulary
    loss = F.cross_entropy(logits, Y if ix is None else Y[ix])
    return loss

# When I ran this locally, I got an optimal learning rate of around 10**-0.8, which is what I set above.
def plot_learning_rates(X, Y):
    lri = []
    lossi = []
    epochs = 1000
    lre = torch.linspace(-3, 0, epochs)
    lrs = 10**lre
    for i in range(epochs):
        # Find a minibatch of data points to train on:
        ix = torch.randint(0, X.shape[0], (minibatch_size,))
        # Forward pass
        loss = forward(X, Y, ix)
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # Update
        lr = lrs[i]
        for p in parameters:
            p.data += -lr * p.grad
        lri.append(lre[i].item())
        lossi.append(loss.item())
    plt.plot(lri, lossi)

def train():
    loss = train_epoch(Xtr, Ytr, iterations=10000, learning_rate=0.8)
    print(f"Loss 1: {forward(Xtr, Ytr).item()}")
    loss = train_epoch(Xtr, Ytr, iterations=10000, learning_rate=0.8)
    print(f"Loss 2: {forward(Xtr, Ytr).item()}")
    loss = train_epoch(Xtr, Ytr, iterations=10000, learning_rate=0.8)
    print(f"Loss 3: {forward(Xtr, Ytr).item()}")
    loss = train_epoch(Xtr, Ytr, iterations=10000, learning_rate=0.08)
    print(f"Loss 4: {forward(Xtr, Ytr).item()}")
    loss = train_epoch(Xtr, Ytr, iterations=10000, learning_rate=0.08)
    print(f"Loss 5: {forward(Xtr, Ytr).item()}")
    print("END Training.")

def train_epoch(X, Y, iterations, learning_rate):
    for _ in range(iterations):
        # Find a minibatch of data points to train on:
        ix = torch.randint(0, X.shape[0], (minibatch_size,))
        # Forward pass
        loss = forward(X, Y, ix)
        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # Update
        for p in parameters:
            p.data += -learning_rate * p.grad
    return loss

    # We use the development set to evaluate the loss, since we don't want to report the training loss (which is biased by overfitting).
    print(f"Development set Loss: {forward(Xdev, Ydev).item()}")
    print(f"Training set Loss: {forward(Xtr, Ytr).item()}")