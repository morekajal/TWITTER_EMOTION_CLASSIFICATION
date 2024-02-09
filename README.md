Emotion Classification using HuggingFace and Distlilbert Transformer with the Twitter 'Emotion' dataset.
Problem Statement : Classify the 'emotion' dataset with the labels provided along into different emotion classes as provided ('sadness', 'joy', 'love', 'anger', 'fear', 'surprise')

Dataset : 
- https://huggingface.co/datasets/dair-ai/emotion/viewer/split/train
- The dataset contains arround 16000 with the columns 'text' (tweets from twitter) and it's 'label'

Data Analysis:
- As you can see in the notebook, the dataset is very much imbalanced. We have used Transformer approach to deal with this imbalance problem.

Tokenization of Dataset:
- For transformers, the input should be a token
- AutoTokenizer : With HF we need the same tokenizer model that we are going to use to build the model. AutoTokenzer solves the issue of finding the perfect model for tokenizer as  per the training model. 
- Understand [CLS] and [SEP] token

Data Encoding :
- text converted to tokens is now converted to encoded vectors

Model Embeddings :
- Used DistilBertForSequenceClassification pretrained-model with the last.hidden_state to generate the embeddings 

Model Building , Training and Evaluation: 
- Used "distilbert-finetuned-emotion" for training

Inferencing:
- For 10 epochs, the test score we got is 92%
- You can play with the training parameters for further improvements.
