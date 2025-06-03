from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences to be embedded
sentences = [
    "This is an example sentence.",
    "Each sentence is converted to a vector."
]

# Generate embeddings
embeddings = model.encode(sentences)

print(embeddings)