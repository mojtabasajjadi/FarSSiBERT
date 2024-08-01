
## FarSSiBERT: A Novel Transformer-based Model for Semantic Similarity Measurement of Persian Social Networks Informal Texts.


FarSSiBERT is a monolingual large language model based on Google’s BERT architecture. This model is pre-trained on large informal Persian short texts with various writing styles from numerous subjects with more than 104M tweets.
Paper presenting FarSSiBERT: <br>
- DOI: https://doi.org/10.48550/arXiv.2407.19173

## Features
It included a Python library for measuring the semantic similarity of Persian short texts.
- Text cleaning.
- Specific tokenizer for informal Persian short text.
- Word and sentence embeddings based on transformers.
- Semantic Similarity Measurement.
- Pre-train BERT model for all downstream tasks especially informal texts.

## How to use:
+ Download and install the FarSSiBERT python package.
+ Import and use as below:

```python
  from FarSSiBERT.SSMeasurement import SSMeasurement

  text1 = "متن اول"
  text2 = "متن دوم"

  new_instance = SSMeasurement( text1 , text2 )
  label = new_instance.get_similarity_label()
  similarity = new_instance.get_cosine_similarity()

```

## Requirements:<br>
  + pyhton=>3.7
  + transformers==4.30.02
  + torch==1.13.0
  + scikit-learn==0.21.3
  + numpy~=1.21.6
  + sklearn~=0.0
