import os
import spacy

from task2.train_ner import train_ner
from task2.utils.paths import NER_MODEL_PATH


def inference_ner(text):
    # check if model exists
    if not os.path.exists(NER_MODEL_PATH):
        # train new model
        nlp = train_ner()
    else:
        # load existing model
        nlp = spacy.load(NER_MODEL_PATH)
    doc = nlp(text)
    return doc


def pretty_print_doc(doc):
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    if not doc.ents:
        print('No entities found')
