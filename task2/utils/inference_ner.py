from pathlib import Path
import spacy

from task2.utils.train_ner import train_ner
from task2.utils.paths import NER_MODEL_PATH


def inference_ner(text):
    # check if model exists
    if not Path(NER_MODEL_PATH).exists():
        print('Model does not exist')
        print('Training a model')
        # train new model
        nlp = train_ner()
    else:
        print(f'Loading model at {NER_MODEL_PATH}')
        # load existing model
        nlp = spacy.load(NER_MODEL_PATH)
    docs = nlp(text)
    return docs


def pretty_print_doc(doc):
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    if not doc.ents:
        print('No entities found')


input_text = input('Enter a sentence for inference: ')
doc = inference_ner(input_text)
pretty_print_doc(doc)
