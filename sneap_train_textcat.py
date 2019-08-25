# https://github.com/explosion/spaCy/blob/master/examples/training/train_textcat.py
#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training
Compatible with: spaCy v2.0.0+
"""
import pandas as pd


from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding


# import data from read_sneap_urls.py
df_pages_all = pd.read_pickle("datasets/df_pages_all_backup.pkl")
#df_pages_all['section_number'] = np.int64(df_pages_all['section_number'])

# Set training data size
train_pct = .8
train_size = round(len(df_pages_all) * train_pct)

#############MAYBE NEED TO CLEAN TEXT DATA MORE??

# randomize
df_pages_all = df_pages_all.sample(frac=1).reset_index()

# Initialize dictionary
df_pages_all['labels'] = None

# Update dictionary by section_name
def update_forum_dict(forum_name, df, iteration):
    if df['section_name'][iteration] == forum_name :
        df['labels'][iteration]['cats'][forum_name ] = True

# create dictionary of labels
for i in df_pages_all.index:

    # default to False    
    df_pages_all['labels'][i] = {'cats': {'Main': False,
     'Backstage': False,
     'FOH': False,
     'Practice_Room': False,
     'Backline': False,
     'Merch_Stand': False,
     'Bar': False }}
    
    # Set dictionary to True for correct forum
    update_forum_dict('Main', df_pages_all, i)
    update_forum_dict('Backstage', df_pages_all, i)
    update_forum_dict('FOH', df_pages_all, i)
    update_forum_dict('Practice_Room', df_pages_all, i)
    update_forum_dict('Backline', df_pages_all, i)
    update_forum_dict('Merch_Stand', df_pages_all, i)
    update_forum_dict('Bar', df_pages_all, i)


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path)
)


df_pages_all['labels'][100]
df_pages_all['section_name'][100]

# Split train/test
df_pages_train = df_pages_all.iloc[0:train_size,:]
df_pages_test = df_pages_all.iloc[train_size:,:]

n_iter=2 
n_texts=2000
    
nlp = spacy.blank("en")  # create blank Language class

# add the text classifier to the pipeline if it doesn't exist
# nlp.create_pipe works for built-ins that are registered with spaCy
if "textcat" not in nlp.pipe_names:
    textcat = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "simple_cnn",
        }
    )
    nlp.add_pipe(textcat, last=True)

# otherwise, get it, so we can add labels to it
else:
    textcat = nlp.get_pipe("textcat")

# add label to text classifier
textcat.add_label("Main")
textcat.add_label("Backstage")
textcat.add_label("FOH")
textcat.add_label("Practice_Room")
textcat.add_label("Backline")
textcat.add_label("Merch_Stand")
textcat.add_label("Bar")

# Create Training set
train_data = list(zip(df_pages_train['initial_message_text'], df_pages_train['labels']))
#train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    if init_tok2vec is not None:
        with init_tok2vec.open("rb") as file_:
            textcat.model.tok2vec.from_bytes(file_.read())
    print("Training the model...")
    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
    batch_sizes = compounding(4.0, 32.0, 1.001)
    for i in range(n_iter):
        losses = {}
        # batch up the examples using spaCy's minibatch
        random.shuffle(train_data)
        batches = minibatch(train_data, size=batch_sizes)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
        print(
            "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                losses["textcat"],
                scores["textcat_p"],
                scores["textcat_r"],
                scores["textcat_f"],
            )
        )

    # test the trained model
    test_text = "This movie sucked"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def load_data(limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    plac.call(main)