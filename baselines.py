# -*- coding: utf-8 -*-

import numpy as np
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
        SGDClassifier(loss='log_loss', penalty='l2', class_weight="balanced",
            early_stopping=True, n_jobs=-1, random_state=42,
        )
    )

def define_baselines():
    pipe1 = Pipeline([
        ("unigram", CountVectorizer(ngram_range=(1,2), max_features=5000)),
        ("LogReg", get_classifier()),
    ])

    pipe2 = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
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
    test_ood_nolabel = evaluate.read_file(evaluate.TEST_NOLABEL_OOD_DATA_PATH)
    test_ids = test_nolabel["id"].values
    test_ood_ids = test_ood_nolabel["id"].values

    texts_train = train_data["text"].values
    texts_test = test_nolabel["text"].values
    texts_test_ood = test_ood_nolabel["text"].values
    texts_train = preprocess_text(texts_train, nlp)
    texts_test = preprocess_text(texts_test, nlp)
    texts_test_ood = preprocess_text(texts_test_ood, nlp)

    scores = []
    for subtask in ["A", "B"]:
        label_selector = evaluate.get_label_selector(subtask)
        labels_train = train_data[label_selector].values

        baselines = define_baselines()
        for baseline in baselines:
            baseline.fit(texts_train, labels_train)
            # predictions on in-domain test set
            predictions = baseline.predict(texts_test)
            score, _ = evaluate.evaluate_predictions(predictions, test_ids, subtask=subtask)
            scores.append([baseline.steps[0][0], subtask, False, score])

            # export predictions to check format
            evaluate.export_predictions(test_ids, predictions, label_selector,
             filename=f"preds_baseline_{baseline.steps[0][0]}-{subtask}.csv")

            # predictions on out-of-domain test set 
            predictions = baseline.predict(texts_test_ood)
            score, _ = evaluate.evaluate_predictions(predictions, test_ood_ids, subtask=subtask, ood=True)
            scores.append([baseline.steps[0][0], subtask, True, score])

            evaluate.export_predictions(test_ood_ids, predictions, label_selector,
             filename=f"preds_baseline_{baseline.steps[0][0]}-{subtask}-ood.csv")
        
        # random baseline on in-domain test set
        predictions = np.random.randint(2, size=(len(texts_test), labels_train.shape[1]))
        score, _ = evaluate.evaluate_predictions(predictions, test_ids, subtask=subtask)
        evaluate.export_predictions(test_ids, predictions, label_selector,
         filename=f"preds_baseline_random-{subtask}.csv")
        scores.append(["random-baseline", subtask, False, score])

        # random baseline on ood test set
        predictions = np.random.randint(2, size=(len(texts_test_ood), labels_train.shape[1]))
        score, _ = evaluate.evaluate_predictions(predictions, test_ood_ids, subtask=subtask, ood=True)
        evaluate.export_predictions(test_ood_ids, predictions, label_selector,
         filename=f"preds_baseline_random-{subtask}-ood.csv")
        scores.append(["random-baseline", subtask, True, score])

    scores = pd.DataFrame(scores, columns=["baseline", "subtask", "ood", "f1-macro"])
    print(pd.DataFrame(scores))

if __name__ == "__main__":
    main()