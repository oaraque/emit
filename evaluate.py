# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from glob import glob
from datetime import datetime
from sklearn.metrics import f1_score, classification_report

TRAIN_DATA_PATH = "data/emit_train.csv"
TEST_DATA_PATH = "data/emit_test.csv"
TEST_OOD_DATA_PATH = "data/emit_test_ood.csv"
TEST_NOLABEL_DATA_PATH = "data/emit_test_nolabel.csv"
TEST_NOLABEL_OOD_DATA_PATH = "data/emit_test_ood_nolabel.csv"

LABELS = [
    'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love',
    'Neutral', 'Sadness', 'Surprise', 'Trust', 'Direction', 'Topic'
]
LABELS_EMOTION = [
    'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Love',
    'Neutral', 'Sadness', 'Surprise', 'Trust', 
]
LABELS_DIRECTION = [
    'Direction', 'Topic',
]

REPORTS_PATH = "reports/"

def check_path(path):
    if not os.path.exists(path):
        raise OSError(f"path {path} not found")

def read_file(path):
    return  pd.read_csv(path)

def read_predictions(path):
    check_path(path)
    
    preds_df = pd.read_csv(path, index_col=0)
    return preds_df

def export_predictions(ids, predictions, labels, path="predictions/", filename=None):
    check_path(path)

    if filename is None:
        now = datetime.strftime(datetime.now(), format="%Y-%m-%d_%H-%M-%S")
        filename = f"preds_{now}.csv"
    preds_df = pd.DataFrame(index=ids, columns=labels, data=predictions)
    preds_df.index.name = "id"
    final_path = os.path.join(path, filename)
    preds_df.to_csv(final_path)
    return final_path


def save_report(report, identifier="submission"):
    check_path(REPORTS_PATH)
    path = os.path.join(REPORTS_PATH, f"{identifier}.txt")
    with open(path, "w") as f:
        f.write(report)

def get_label_selector(subtask):
    label_selector = LABELS
    if subtask == "A":
        label_selector = LABELS_EMOTION
    elif subtask == "B":
        label_selector = LABELS_DIRECTION
    else: # a default case, considering all labels
        label_selector = LABELS
    return label_selector

def evaluate_predictions(predictions, predictions_ids, subtask="A", ood=False):
    label_selector = get_label_selector(subtask)
    if not ood:
        labels_gold = read_file(TEST_DATA_PATH)[label_selector]
        ids_gold = read_file(TEST_DATA_PATH)["id"]
    else:
        labels_gold = read_file(TEST_OOD_DATA_PATH)[label_selector]
        ids_gold = read_file(TEST_OOD_DATA_PATH)["id"]
    
    # check that prediction and gold set IDs are the same
    assert (predictions_ids == ids_gold).all()

    score = f1_score(labels_gold, predictions, average="macro")
    clf_report = classification_report(labels_gold, predictions, target_names=label_selector, digits=4)
    clf_report_dict = classification_report(labels_gold, predictions, target_names=label_selector,
                                            digits=4, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report_dict)

    values_export = clf_report_df.loc["f1-score", label_selector].T.values
    values_export = "\t".join([str(round(v,4)) for v in values_export])

    report = f"""
    F1-macro: {score}
    {clf_report}
    """
    print(report)
    print(values_export)
    return score, report

def check_predictions_format(preds, subtask):
    label_selector = get_label_selector(subtask)
    assert (preds.columns == label_selector).all()
    assert preds.index.name == "id"

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions for EMit at EVALITA")
    parser.add_argument("predictions_file", type=str, help="Predictions file to evaluate")
    parser.add_argument("task", choices=["A", "B"], help="task for wich evaluate, can be A or B")
    parser.add_argument("--glob", action="store_true",
     help="If specified, the arg predictions_file contains a glob pattern for obtaining several predictions files")
    parser.add_argument("--ood", action="store_true",
        help="If specifed, Out of Domain (OOD) evaluation must be done")
    args = parser.parse_args()

    # read predictions
    if args.glob:
        preds_files = glob(args.predictions_file)
    else:
        preds_files = [args.predictions_file, ]

    for preds_file in preds_files:
        preds_id = os.path.splitext(os.path.basename(preds_file))[0]
        preds_input = read_predictions(preds_file)
        preds_ids = pd.Series(preds_input.index.values)
        check_predictions_format(preds_input, args.task)
        preds_input = preds_input[get_label_selector(args.task)]

        # evaluate predictions
        score, report = evaluate_predictions(preds_input, predictions_ids=preds_ids, subtask=args.task, ood=args.ood)
        save_report(report, identifier=preds_id)
        print(f"Predictions '{preds_id}' score: {score}")


if __name__ == "__main__":
    main()