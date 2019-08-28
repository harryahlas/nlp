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
import numpy as np

from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding

import operator # for evaluation

# import data from read_sneap_urls.py
df_pages_all = pd.read_pickle("datasets/df_pages_all_backup.pkl")
#df_pages_all['section_number'] = np.int64(df_pages_all['section_number'])


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


df_pages_all['labels'][30100]
df_pages_all['section_name'][30100]

# Remove NAs
df_pages_all.dropna(subset = ['initial_message_text'], inplace = True)

# Set training data size
train_pct = .8
train_size = round(len(df_pages_all) * train_pct)

# Split train/test
df_pages_train = df_pages_all.iloc[0:train_size,:]
df_pages_test = df_pages_all.iloc[train_size:,:].reset_index()

#dfx = df_pages_all[df_pages_all.initial_message_text.isnull()]
#dfx = df_pages_all.dropna(subset = ['initial_message_text'], inplace = True)
                    
n_iter=10 
#n_texts=2000
    
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

### Evaluation function (needs work)
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

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
init_tok2vec=None
# Train
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
            
            if texts[0] == None or annotations[0] == None:
                print('texts[0]  | annotations[0] == None, continuing')
                continue
            
            nlp.update(texts, 
                       annotations, 
                       sgd=optimizer, 
                       drop=0.2, 
                       losses=losses
                       )
#        with textcat.model.use_params(optimizer.averages):
#            # evaluate on the dev data split off in load_data()
#            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
#        print(
#            "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
#                losses["textcat"],
#                scores["textcat_p"],
#                scores["textcat_r"],
#                scores["textcat_f"],
#            )
#       )



df_pages_train.initial_message_text[30045]
df_pages_train.labels[30045]
df_pages_train.initial_message_text[30044] == None
df_pages_train.labels[30044]


# import from drive
nlp = pickle.load(open("models/nlp_sneap.pickle", "rb"))


# test the trained model
test_text = "What do you think about the distressor? I tried it out for the first time. NOt sure how to use it though. Help!"
doc = nlp(test_text)
print(test_text, doc.cats)

test_row_num = 127
test_text = df_pages_test.iloc[test_row_num]['initial_message_text']
doc = nlp(test_text)
print(test_text, 
      doc.cats,
      "Correct section: " + df_pages_test.iloc[test_row_num]['section_name'])

 
# Evaluate on test data
df_pages_test['recommendation'] = str()
df_pages_test['certainty'] = float()

for i in df_pages_test.index:
    print(i)
    test_text = df_pages_test.iloc[i]['initial_message_text']
    doc = nlp(test_text)
    prediction = max(doc.cats.items(), key=operator.itemgetter(1))
    df_pages_test['recommendation'][i] = prediction[0]
    df_pages_test['certainty'][i] = prediction[1]

df_pages_test['recommendation'].value_counts()    
df_pages_test['correct'] = np.where(df_pages_test['recommendation'] == df_pages_test['section_name'], 'correct', 'incorrect')
df_pages_test['correct'].value_counts()

df_pages_test.groupby(['correct', 'recommendation']).mean()['certainty']

# Save model
nlp.to_disk("nlp20190824")
nlp2 = spacy.load("nlp20190824")
    
## Optional save, needs work
if output_dir is not None:
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)
    print(test_text, doc2.cats)

