import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example

TRAIN_DATA = [
    ("Seller: Alpha Exports Pvt. Ltd. Buyer: Beta Imports LLC Total Cost: $2400", {
        "entities": [
            (8, 31, "SELLER"),         
            (39, 56, "BUYER"),          
            (70, 76, "TOTAL_COST")      
    }),
    ("Exporter: Alpha Exports Pvt. Ltd. Importer: Beta Imports LLC Total Weight: 350 kg", {
        "entities": [
            (10, 33, "EXPORTER"),
            (44, 61, "IMPORTER"),
            (77, 83, "TOTAL_WEIGHT")
        ]
    }),
    ("Consignor: ABC Ltd. Consignee: XYZ Co. Weight: 200 kg", {
        "entities": [
            (11, 19, "CONSIGNOR"),
            (31, 38, "CONSIGNEE"),
            (47, 54, "WEIGHT")
        ]
    })
]

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

for _, annotations in TRAIN_DATA:
    for start, end, label in annotations["entities"]:
        ner.add_label(label)

doc_bin = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    doc_bin.add(example.reference)

doc_bin.to_disk("training_data.spacy")

from spacy.training import Example
import random

optimizer = nlp.begin_training()
for i in range(30):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], losses=losses, drop=0.2, sgd=optimizer)
    print(f"Iteration {i+1} Losses: {losses}")

text = "Buyer: Moon Co. Seller: Sunray Ltd. Total Cost: $5000"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

nlp.to_disk("custom_ner_model")
