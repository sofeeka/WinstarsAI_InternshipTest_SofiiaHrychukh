import os
import spacy
from pathlib import Path

from task2.utils.train_ner import train_ner
from task2.utils.paths import NER_MODEL_PATH


def inference_ner(text):
    # check if model exists
    if not os.path.exists(NER_MODEL_PATH):
        print('Model does not exist')
        print('Training a model')
        # train new model
        nlp = train_ner()
    else:
        print(f'Loading model at {NER_MODEL_PATH}')
        # load existing model
        nlp = spacy.load(NER_MODEL_PATH)
    doc = nlp(text)
    return doc


def pretty_print_doc(doc):
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    if not doc.ents:
        print('No entities found')
