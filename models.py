import transformers, pandas as pd, sklearn, torch, numpy as np
from transformers import BertTokenizer, BertModel, DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class BaseRatingsClassifier:
    """Abstract base class for ratings classifiers."""
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        pass

    def predict(self, input_sentence):
        return None

    def compute_accuracy(self, X_test, y_test):
        num_correct = 0
        for x, y in zip(X_test, y_test):
            pred = self.predict(x)
            if pred == y:
                num_correct += 1
        return num_correct/len(X_test)


class LogitRegOnBertEmb(BaseRatingsClassifier):
    """Logistic regression classifier on Bert embeddings."""
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = LogisticRegression(random_state=0)

    def compute_Bert_embedding(self, X):
        # Compute Bert embedding of datum X, pad/truncate as neccesary:
        inputs = self.tokenizer(X, return_tensors="pt", padding='max_length',
                truncation=True, max_length=512)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states.detach().numpy()
        return last_hidden_states.flatten()

    def compute_Bert_embeddings(self, data):
        # Transform data into Bert embeddings:
        embeddings = []
        for d in data:
            # Append the flattened Bert embedding to data:
            last_hidden_states = self.compute_Bert_embedding(d)
            embeddings.append(last_hidden_states)

        return np.array(embeddings)

    def train(self, X_train, y_train):
        # Transform X_train into Bert embeddings:
        X_train_embeddings = self.compute_Bert_embeddings(X_train)

        # Train classifier:
        self.classifier.fit(X_train_embeddings, y_train)

    def predict(self, input_sentence):
        embedding = self.compute_Bert_embedding(input_sentence)
        ret = self.classifier.predict(np.expand_dims(embedding, axis=0))
        return ret[0] # ret is either [0] or [1]

    def compute_accuracy(self, X_test, y_test):
        X_test_embeddings = self.compute_Bert_embeddings(X_test)
        return self.classifier.score(X_test_embeddings, y_test)


class DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for given lists of encodings/labels. Can be used as a torch Dataset object."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class FinetuningBert(BaseRatingsClassifier):
    """Classifier trained by finetuning on distilBert."""
    def __init__(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.trainer = None

    def train(self, X_train, y_train):
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True)
        train_dataset = DatasetWrapper(train_encodings, y_train)
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )
        
        self.trainer = Trainer(
            model=self.model,                # the instantiated model to be trained
            args=training_args,              # training arguments, defined above
            train_dataset=train_dataset,     # training dataset
        )

        self.trainer.train()

    def predict(self, input_sentence):
        inputs = self.tokenizer(input_sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        if outputs['logits'][0][0] > outputs['logits'][0][1]:
            return 0
        else:
            return 1

    def compute_accuracy(self, X_test, y_test):
        # Note: one can also use the function inherited from the base class; however,
        # using the built-in prediction method, as is done here, seems much faster.
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True)
        test_dataset = DatasetWrapper(test_encodings, y_test)
        predictions = self.trainer.predict(test_dataset) # Compute predictions
        preds = np.argmax(predictions.predictions, axis=-1) # Transform output to labels
        return sklearn.metrics.accuracy_score(preds, np.array(y_test))

    def print_config(self):
        print(self.model.config)


class OutOfBoxBert(BaseRatingsClassifier):
    """Wrapper for an out-of-box distilBert classifier."""
    def __init__(self):
        self.sentiment = transformers.pipeline(task='sentiment-analysis')

    def predict(self, input_sentence):
        # DistilBert has an input max token length of 512, so opt for crude 
        # truncation for now:
        input_sentence = input_sentence[:512] 
        result = self.sentiment(input_sentence)
        label = 0 if result[0]['label'] == 'NEGATIVE' else 1
        return label

    def print_config(self):
        print(self.sentiment.model.config)


def main():
    d = pd.read_csv('data/reviews_Baby_5_preprocessed.csv')
    d = d.dropna() # Some data have no review (i.e., NaN valued)
    #d = d[:10] # NOTE: Only 10 data for test purposes
    X_train, X_test, y_train, y_test = train_test_split(d['review'],
            d['label'], test_size=0.33, random_state=10)

    # Convert to lists so it's easier to deal with (pandas series is unwieldy):
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()

    # Subsample training data to speed up training. Comment out the line below to train on full data.
    X_train, y_train = X_train[:500], y_train[:500]

    print('Training on a set of {} reviews.'.format(len(X_train)))
    print('Testing on a set of {} reviews.'.format(len(X_test)))

    LRonBE = LogitRegOnBertEmb()
    print('Fitting logistic regression on Bert embeddings...')
    LRonBE.train(X_train, y_train)
    print('Evaluating Bert-based LR classifier on the test set...')
    acc = LRonBE.compute_accuracy(X_test, y_test)
    print('Bert-based LR\'s accuracy is {}.'.format(acc))

    finetune = FinetuningBert()
    print('Finetuning on Bert...')
    finetune.train(X_train, y_train)
    print('Evaluating fine-tuned Bert on the test set...')
    acc = finetune.compute_accuracy(X_test, y_test)
    print('Fine-tuned Bert\'s accuracy is {}'.format(acc))

    outofbox = OutOfBoxBert()
    print('Evaluating out-of-box classifier on the test set...')
    acc = outofbox.compute_accuracy(X_test, y_test)
    print('Out-of-box\'s accuracy is {}.'.format(acc))


if __name__ == "__main__":
    main()
