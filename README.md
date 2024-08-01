
## FarSSiBERT: A Pre-trained Language Model for Semantic Similarity Measurement of Persian Informal Short Texts.


FarSSiBERT is a monolingual large language model based on Google’s BERT architecture. This model is pre-trained on large informal Persian short texts with various writing styles from numerous subjects with more than 104M tweets.
Paper presenting FarSSiBERT: DOI:https://doi.org/10.48550/arXiv.2407.19173

## Features
It included a Python library for measuring the semantic similarity of Persian short texts.
+ Text cleaning.
+ Specific tokenizer for informal Persian short text.
+ Word and sentence embeddings based on transformers.
+ Semantic Similarity Measurement.
+ Pre-train BERT model for all downstream tasks especially informal texts.

## How to use:
+ Download and install the FarSSiBERT python package.
+ Import and the use it:

```python
  from FarSSiBERT.SSMeasurement import SSMeasurement
  <br>
  text1 = "متن اول"
  <br>
  text2 = "متن دوم"
  <br>
  new_object = SSMeasurement( text1 , text2 )<br>
  label = new_object.get_similarity_label()<br>
  similarity = new_object.get_cosine_similarity()<br>
  <br>
```

## Requirements:<br>
  + pyhton=>3.7<br>
  + transformers==4.30.02<br>
  + torch==1.13.0<br>
  + scikit-learn==0.21.3<br>
  + numpy~=1.21.6<br>
  + sklearn~=0.0<br>
