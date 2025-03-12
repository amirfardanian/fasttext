from capture_model.modelling import Model
from shutil import copyfile
import csv
import json
import pandas as pd
import os


class LookUpClassifierModel(Model):

    def __init__(self, model_name, storage_dir=''):
        super().__init__(model_name)
        self.storage_dir = storage_dir + '/' if storage_dir else ''
        self.look_up_table = {}  # ✅ Changed from `None` to `{}` to avoid errors
        self.look_up_csv_filename = None
        self.missing_label = "-1"

    def train(self, csv_filepath, input_names, output_names, missing_label="-1", *args, **kwargs):
        if len(input_names) > 1 or len(output_names) > 1:
            raise ValueError('LookupModel can only train on length one input and output list.')

        filename = os.path.basename(csv_filepath)
        lookup_csv_storage_path = os.path.join(self.storage_dir, filename)

        try:
            copyfile(csv_filepath, lookup_csv_storage_path)
        except Exception as e:
            self.logger.warning(f'⚠️ Exception copying CSV file: {e}')

        self.look_up_csv_filename = filename
        self.missing_label = missing_label
        self.input_names = input_names
        self.output_names = output_names
        self._load_csv_to_dict(lookup_csv_storage_path)  # ✅ Ensure correct file path
        return 1

    def _load_csv_to_dict(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.storage_dir, self.look_up_csv_filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                self.look_up_table = {
                    row[self.input_names[0]].strip().lower(): row[self.output_names[0]] 
                    for row in csv_reader
                }
            print(f"✅ Lookup model loaded {len(self.look_up_table)} entries from {filepath}")
        except Exception as e:
            print(f"❌ Failed to load lookup data: {e}")
            self.look_up_table = {}

    def predict_one(self, item, incl_probabilities=False, *args, **kwargs):
        item = item.strip().lower()  # ✅ Standardizing input for lookup

        print(f"🔍 Searching for: '{item}' in lookup table...")  # Debugging print

        if incl_probabilities:
            prediction = self.look_up_table.get(item, self.missing_label)
            score = 1.0 if prediction != self.missing_label else 0.0
            result = [(prediction, score)]
        else:
            result = [self.look_up_table.get(item, self.missing_label)]

        print(f"✅ Lookup result for '{item}': {result}")  # Debugging print
        return result

    def predict_many(self, items, incl_probabilities=False, *args, **kwargs):
        if isinstance(items, pd.DataFrame):
            items = items[self.input_names[0]]
            result = self._predict_series(items)
        elif isinstance(items, pd.Series):
            result = self._predict_series(items)
        else:
            result = [self.predict_one(item, incl_probabilities) for item in items]

        return result

    def _predict_series(self, items, **kwargs):
        output_name = None if len(self.output_names) == 0 else self.output_names[0]
        result = [self.predict_one(item, **kwargs) for item in items]
        return pd.Series(result, name=output_name, index=items.index)

    def to_json(self):
        filepath = self.storage_dir + self.model_name + '.json'
        model_info_dict = self._to_model_info()

        with open(filepath, 'w') as file:
            json.dump(model_info_dict, file)

    def _to_model_info(self):
        super_model_info = super()._to_model_info()
        model_info = {
            **super_model_info,
            'storage_dir': self.storage_dir,
            'look_up_csv_filename': self.look_up_csv_filename,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'init_datetime': self.init_datetime.strftime(self.datetime_format),
            'missing_label': self.missing_label
        }
        return model_info

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, 'r') as json_file:
            model_info_dict = json.load(json_file)
        model = cls._from_model_info(model_info_dict)

        if model.look_up_csv_filename:
            model_dir = os.path.dirname(filepath)
            lookup_filepath = os.path.join(model_dir, model.look_up_csv_filename)
            model._load_csv_to_dict(lookup_filepath)

        return model

    @classmethod
    def _from_model_info(cls, model_info_dict):
        model = cls(model_info_dict['model_name'])

        model_info_keys = model._to_model_info().keys()
        for attr_name, attr_value in model_info_dict.items():
            if attr_name in model_info_keys:
                setattr(model, attr_name, model._handle_model_attr_value(attr_name, attr_value))
            else:
                model.logger.warning(f"⚠️ Unknown attribute: {attr_name}")

        return model

    # ✅ **Fix: Implement missing abstract methods**
    def get_labels(self):
        """Return a sorted list of unique labels from the lookup table."""
        return sorted(set(self.look_up_table.values()))

    def predict_proba(self, items, **kwargs):
        """Return a list of (label, probability) tuples for given items."""
        return [(self.predict_one(item), 1.0) for item in items]
