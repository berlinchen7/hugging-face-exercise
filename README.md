# Hugging Face Exercise
This is an exercise in interacting with Hugging Face. The task is to train/evaluate various Bert-based models on a chosen dataset of Amazon reviews. In particular, we took the set of reviews for baby products (download the dataset [here](https://jmcauley.ucsd.edu/data/amazon/)), preprocessed so that reviews are binary (see `preproc_utils.py` for details), split into train/test sets, and compute the accuracy score of 3 (trained) Bert-based models. The models are as follows:

1. Logistic regression classifier trained on Bert embeddings (abbv. as **LRonBE**).

2. Fine-tuning the dataset on a pertrained DistilBert model with a single dense layer as the head (abbv. as **finetune**) . This can be seen as a deep classifier for the task.

3. An out-of-box DistilBert-based classifier (abbv. as **outofbox**) that was originally constructed by fine-tuning a pertrained DistilBert model on the SST2 dataset.

Based on a sample of 500[^1] training data and 3299 testing data, we obtained following the accuracy scores:
|Model  |LRonBE  |finetune  |outofbox  |
|--|--|--|--|
|Accuracy  |0.88  |0.87  |0.79  |

[^1]:  The size of the original training data is 6697; however, we did not have enough computational resource to fine-tune the model on the full training dataset (e.g., by extrapolation, we estimated that it would take at least 15 hours to finish training **finetune** on our machine, which is a 2-core Intel i5 2.7 GHz laptop). Hence, we only trained the model on a subset of size 500 (however, we still evaluated the model on the full testing data which has size 3299).

To reproduce the preprocessed data (`./data/reviews_Baby_5_preprocessed.csv`), clone this repository, download/unzip the dataset by clicking [this link](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz), move it to the `./data` directory, and then run `python preproc_utils.py` after downloading the necessary packages. The preprocessed csv file will be in the `./data` directory.

To reproduce the accuracy scores, simply clone this repository and run `python models.py` after downloading the necessary packages. Note that this will take a considerable amount of time.
