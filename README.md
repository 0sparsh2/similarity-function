# similarity-function

In the gives codes, there are two version. One using Pytorch converting inputs as tensors and then evaluting using the cosine similarity while the other method uses sklearn library with inputs as 2D arrays and then finding the cosine similarity between the two. 

Run this in terminal to install dependencies
```pip install -r requirements.txt```

To avoid installing sentence-transformers model every single time, run the following command in terminal
```python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('bert-base-nli-mean-tokens')"```
