# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from gsitk.preprocess import pprocess_twitter, Preprocessor
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
import spacy

import evaluate

def get_classifier():
    return MultiOutputClassifier(
        SGDClassifier(
            loss='log_loss', penalty='l2', class_weight="balanced",
            early_stopping=True, n_jobs=-1, random_state=42,
        )
    )

def define_baselines():
    pipe1 = Pipeline([
        ("unigram", CountVectorizer(ngram_range=(1,2))),
        ("LogReg", get_classifier()),
    ])

    pipe2 = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("LogReg", get_classifier()),
    ])

    return [pipe1, pipe2]

def preprocess_text(texts, nlp):
    texts = Preprocessor(pprocess_twitter).fit_transform(texts)
    texts = [str(text) for text in texts]
    docs = nlp.pipe(texts)

    tokens_docs = []
    for doc in tqdm(docs):
        tokens = []
        for token in doc:
            tokens.append(token.text.lower())
        tokens_docs.append(tokens)
    
    return [" ".join(doc) for doc in tokens_docs]

def main():
    nlp = spacy.load("it_core_news_sm")

    train_data = evaluate.read_file(evaluate.TRAIN_DATA_PATH)
    test_nolabel = evaluate.read_file(evaluate.TEST_NOLABEL_DATA_PATH)
    test_ids = test_nolabel["id"].values

    texts_train = train_data["text"].values
    texts_test = test_nolabel["text"].values
    labels_train = train_data[evaluate.LABELS].values

    texts_train = preprocess_text(texts_train, nlp)
    texts_test = preprocess_text(texts_test, nlp)

    scores = []
    baselines = define_baselines()
    for baseline in baselines:
        baseline.fit(texts_train, labels_train)
        predictions = baseline.predict(texts_test)
        score, _ = evaluate.evaluate_predictions(predictions)
        scores.append([baseline.steps[0][0], score])

        # export predictions to check format
        evaluate.export_predictions(test_ids, predictions)

    scores = pd.DataFrame(scores, columns=["baseline", "f1 macro"])
    print(pd.DataFrame(scores))

if __name__ == "__main__":
    main()