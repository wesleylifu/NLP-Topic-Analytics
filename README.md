This project is to build a text classification model that can classify new messages into the given topics to compete in Codalab competition. 

## Data Preprocessing: 
build Custom Functions for Preprocessing and Feature Engineering to achieve accuracy excellence against a manually labled dataset with throusnads of finance messages

## Model Architecture: 
develop the best model that is able to train with high accuracy and without any errors. To achieve the goal, different methodologies were tried with:

- Shallow ML-based: Word Enbedding (TF-IDF) + Boosting (SVM, DT, RF, Boosting...)(Shuyi)

- Deep Learning: Sentence Embedding (word2vec,BERT) + Deep NN(such as CNN, RNN, LSTM, GRU, Transformer...)

- Transfer learning: Few-shot, zero-shot, Pipeline building in Deep Learning 

- Hybrid: Sentence Embedding (word2vec,BERT) + Boosting (KNN, RF, LGBM, XGBoost) 

## Best Performing Model: 
Deep ML Roberta with fine tuning win the competition as our best model. 

## Conclusion:
NLP with transfer learning can achieve superior performance and in terms of the model performance result from competition, in a descending order which is Transfer learning, deep learning, hybrid and shallow. 

## Lesson Learned:
- Removing stopwords and lemmatization could be helpful in the preprocess step, but these steps are not necessary for Transformer models. 
- Word2vec is not easy to implement, not just because most examples and documents found on the internet will have a conflict with the current package version. Still, the model could be tough to train if we use the SoftMax function since there are too many numbers of categories in the label. 
- Training a model from scratch could be hard for high accuracy and time-consuming, while pre-trained models are much easier to use, and so does the pre-trained embedding layer
- Early stopping is essential for hyper-parameter tuning since it will stop at the critical point. 
- Transfer learning is slower than regular deep learning models. 
- In shallow machine earning, MLPClassifier could be the best model to achieve high accuracy in a short period.

## Recommendation:
- With more time, resource it would be highly recommended taking advantage of pre-trained NLP model available in Hugging Face; 
- Broadly employ Ktrain learner to train model and harness its amazing benefit;
- Invest more in data preprocessing as we identify some abnormal has to do with labeling in cosine similarity analysis mentioned before by Andrew. It could be a potential opportunity to improve model performance further;
- Last but not the least, use Optuna to tune NLP model in a one stop shop manner.  

