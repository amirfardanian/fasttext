import sys
sys.path.append('/home/ec2-user/environment/capture_model_old-master')

from capture_model.modelling import Model
import fasttext as ft
import os
import json
import pandas as pd
import numpy as np

class FasttextClassifierModel(Model):

    def __init__(self, model_name, storage_dir=''):
        super(FasttextClassifierModel, self).__init__(model_name)
        self.storage_dir = storage_dir + '/' if storage_dir else ''
        self.internal_model = None
        self.internal_model_filepath = None

    def train(self, dataframe, input_names, output_names, **kwargs):
        """Trains the FastText model and saves it."""
        model_path = self.storage_dir + self.model_name
        dataframe["label"] = "__label__" + dataframe["label"].astype(str)

        training_filepath = self.storage_dir + 'fasttext_training.txt'
        dataframe.to_csv(training_filepath, sep=' ', index=False, header=False)

        try:
            print(" Training FastText model...")
            self.internal_model = ft.train_supervised(input=training_filepath, **kwargs)
            print("✅FastText model training completed!")

            print(f" Saving model to: {model_path}.bin")
            self.internal_model.save_model(model_path + ".bin")

        except Exception as e:
            print(f"❌ FastText training failed: {e}")

        os.remove(training_filepath)
        self.internal_model_filepath = self.model_name + ".bin"

        #  Save model metadata
        self.to_json()
        return 1

    def predict_one(self, item: str, k_best=1, incl_probabilities=False, **kwargs):
        """Predicts a single line."""
        item = item.replace("\n", " ").strip()
        labels, probabilities = self.internal_model.predict(item, k=k_best, **kwargs)
        return list(zip(labels, probabilities)) if incl_probabilities else labels

    def predict_many(self, items, k_best=1, incl_probabilities=False, **kwargs):
        """Predicts multiple lines at once."""
        inputs = [text.replace("\n", " ").strip() for text in items]
        predictions = [self.internal_model.predict(text, k=k_best, **kwargs) for text in inputs]

        if incl_probabilities:
            return predictions
        return [labels for labels, _ in predictions]

    def predict_proba(self, items, **kwargs):
        """Returns label probabilities for multiple receipt lines."""
        inputs = [text.replace("\n", " ").strip() for text in items]
        predictions = [self.internal_model.predict(text, k=-1, **kwargs) for text in inputs]

        all_labels = sorted(self.internal_model.get_labels())
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        probs_matrix = np.zeros((len(inputs), len(all_labels)))

        for i, (labels, probs) in enumerate(predictions):
            for label, prob in zip(labels, probs):
                idx = label_to_idx[label]
                probs_matrix[i, idx] = prob

        return probs_matrix

    def get_labels(self):
        """Returns all possible labels the model can predict."""
        return sorted(self.internal_model.get_labels())

    def _to_model_info(self):
        """Saves model metadata."""
        super_model_info = super(FasttextClassifierModel, self)._to_model_info()
        return {
            **super_model_info,
            'storage_dir': self.storage_dir,
            'internal_model_filepath': self.internal_model_filepath
        }

    def to_json(self):
        """ Saves model metadata to JSON."""
        model_info = {
            'model_name': self.model_name,
            'storage_dir': self.storage_dir,
            'internal_model_filepath': self.model_name + ".bin"
        }
        json_path = os.path.join(self.storage_dir, self.model_name + ".json")

        with open(json_path, 'w') as json_file:
            json.dump(model_info, json_file, indent=4)

        print(f" Model metadata saved at: {json_path}")

    @classmethod
    def from_json(cls, filepath):
        """Loads a FastText model from a JSON file."""
        with open(filepath, 'r') as json_file:
            model_info_dict = json.load(json_file)

        model = cls(model_info_dict['model_name'], storage_dir=model_info_dict['storage_dir'])

        model_path = os.path.join(model.storage_dir, model_info_dict['internal_model_filepath'])
        model.internal_model = ft.load_model(model_path)

        return model
