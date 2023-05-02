# emit
Official repository for [EMit](http://www.di.unito.it/~tutreeb/emit23/index.html) @ [EVALITA](https://www.evalita.it/campaigns/evalita-2023/).

Join the shared task in the GoogleGroup: emit_evalita2023@googlegroups.com

## Guidelines
Important information on the EMit share task, guidelines and details: [EMit_guidelines.pdf](EMit_guidelines.pdf).


## Dataset
You can find the dataset [`release/emit_2023.zip`](release/emit_2023.zip) file.

**NEW!**
Test set released, find it in [`release/emit_2023-test.zip`](release/emit_2023-test.zip).

To obtain the dataset, please fill in the following form: https://forms.gle/6tHUZ4qcwLfkNior9 

## Evaluation script

We provide the **official evaluation script** in the [`evaluate.py`](evaluate.py) file.
This aids to understand the final evaluation of the shared task.

Additionally, the file [`baselines.py`](baselines.py) provides the code for the evaluation of some baselines (unigrams, TF-IDF).
_This is not mandatory_, but you can use this code to better understand the evaluation of the task.

## Submission examples
You can find submissions examples for Subtasks A and B in the [`submission_example_task-A.csv`](submission_example_task-A.csv) and [`submission_example_task-B.csv`](submission_example_task-B.csv).

Please submit your runs in **ZIP format to emit-evalita2023@gmail.com**

### Submission example for Task A
The submission file must be in CSV format, and contain the `id` of the message and all the labels of subtask A:
```
id,Anger,Anticipation,Disgust,Fear,Joy,Love,Neutral,Sadness,Surprise,Trust
246b5a294b3208be6800067b9f0f9e87,1,0,1,0,0,1,0,0,0,0
611f78bf3db6c71f0ed053459889fd4b,0,0,1,0,1,1,1,1,1,0
d37b6e59c8ba22a23755566c7797f69a,0,1,0,1,0,1,1,1,0,1
60cbfcc1eeaed1d8adf8fd7b3bc17dba,1,0,0,1,1,1,0,0,1,0
601019dedf4f3be95e8f601fae0dc820,0,0,1,0,1,0,1,1,0,1
```

### Submission example for Task B
The submission file must be in CSV format, and contain the `id` of the message and all the labels of subtask B:
```
id,Direction,Topic
246b5a294b3208be6800067b9f0f9e87,1,0
611f78bf3db6c71f0ed053459889fd4b,1,0
d37b6e59c8ba22a23755566c7797f69a,0,0
60cbfcc1eeaed1d8adf8fd7b3bc17dba,0,1
601019dedf4f3be95e8f601fae0dc820,1,0
```
