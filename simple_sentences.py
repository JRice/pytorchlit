from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# This will be a simple script to demonstrate how to calculate the distance between sentence embeddings using the SentenceTransformer
# library and cosine similarity. The sentences chosen are meant to be semantically different, so we expect the distance between them
# to reflect that. This was based on https://youtu.be/YDdKiQNw80c?si=69I0IWrti8llzast&t=600

# Load the same model mentioned in the video
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define the three example sentences
sentences = [
    "Why is the sky blue?",
    "The sky is blue due to a process called Rayleigh scattering.",
    "Bicycles typically have two wheels, otherwise they're called a tricycle or a car."
]

# Create embeddings (E1, E2, E3)
embeddings = model.encode(sentences)

# Calculate Cosine Similarity
# Note: The video discusses "distance". In scikit-learn, 1 - similarity = distance.
def get_distance(v1, v2):
    sim = cosine_similarity([v1], [v2])[0][0]
    return 1 - sim

dist_1_2 = get_distance(embeddings[0], embeddings[1])
dist_1_3 = get_distance(embeddings[0], embeddings[2])

print(f"Distance between 'Sky Blue' and 'Rayleigh Scattering': {dist_1_2:.4f}")
print(f"Distance between 'Sky Blue' and 'Bicycles': {dist_1_3:.4f}")
