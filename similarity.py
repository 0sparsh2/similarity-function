from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence1 = input("Enter sentence 1")
sentence2 = input("Entern sentence 2")
sentences = [sentence1,sentence2]

sentence_embeddings = model.encode(sentences)
val = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
print("Similarity between sentences is", val)
