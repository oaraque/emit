import os
from glob import glob
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import evaluate
import baselines

def test_read_data():
    data = evaluate.read_file("data/emit_sample.csv")
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (100, 14)
    assert (data.columns == ['id', 'text', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise', 'Trust', 'Direction', 'Topic']).all()

def test_read_data_test():
    labeled = evaluate.read_file("data/emit_test.csv")
    no_labeled = evaluate.read_file("data/emit_test_nolabel.csv")
    assert (labeled.columns == ['id', 'text', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise', 'Trust', 'Direction', 'Topic']).all()
    assert (no_labeled.columns == ["id", "text"]).all()

def test_read_data_sample():
    for file_i in glob("data/*.csv"):
        data = evaluate.read_file(file_i)
        assert not data.isnull().any().any()

def test_baselines_define_baselines():
    models = baselines.define_baselines()
    assert isinstance(models, list)
    assert len(models) > 0
    assert len(models) == 2
    for model in models:
        assert isinstance(model, Pipeline)

def test_evaluate_evaluate_predictions():
    for subtask in ["A", "B"]:
        dim = 10 if subtask=="A" else 2
        predictions_fake = np.random.randint(2, size=(1000, dim))
        score, report = evaluate.evaluate_predictions(predictions_fake, subtask=subtask)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(report, str)
        assert len(report) > 0

def test_evaluate_export_read_predictions():
    # export predictions
    predictions_fake = np.random.randint(2, size=(1000, 10))
    ids_fake = np.arange(1000)
    path = evaluate.export_predictions(ids_fake, predictions_fake, evaluate.LABELS_EMOTION)

    # read predictions 
    predictions = evaluate.read_predictions(path)
    assert (predictions.values == predictions_fake).all()
    assert (predictions.index.values == ids_fake).all()
    assert (predictions.columns == evaluate.LABELS_EMOTION).all()
    assert predictions.index.name == "id"
    os.remove(path)

def test_evaluate_save_report():
    msg = "My test report"
    evaluate.save_report(msg, "test-report")
    path = os.path.join(evaluate.REPORTS_PATH, "test-report.txt")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    with open(path, "r") as f:
        read = f.read().strip()
    assert read == msg
    os.remove(path)


## Tests about the data files
def test_data_ids():
    train = evaluate.read_file("data/emit_train.csv")
    train_A = evaluate.read_file("data/emit_train_A.csv")
    train_B = evaluate.read_file("data/emit_train_B.csv")
    test = evaluate.read_file("data/emit_test.csv")
    test_nolabel = evaluate.read_file("data/emit_test_nolabel.csv")

    assert len(set(train["id"]) & set(train_A["id"])) == train.shape[0]
    assert len(set(train["id"]) & set(train_B["id"])) == train.shape[0]
    assert len(set(test["id"]) & set(train["id"])) == 0
    assert len(set(test_nolabel["id"]) & set(test["id"])) == test.shape[0]

def test_data_cols():
    train = evaluate.read_file("data/emit_train.csv")
    train_A = evaluate.read_file("data/emit_train_A.csv")
    train_B = evaluate.read_file("data/emit_train_B.csv")

    assert (train.columns == ["id", "text"] + evaluate.LABELS).all()
    assert (train_A.columns == ["id", "text"] + evaluate.LABELS_EMOTION).all()
    assert (train_B.columns == ["id", "text"] + evaluate.LABELS_DIRECTION).all()