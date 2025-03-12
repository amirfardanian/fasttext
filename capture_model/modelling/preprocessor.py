from capture_model.utils import get_console_logger
from capture_model.modelling.transformations import StatelessTransformation
from capture_model.modelling.string_functions import *
import hashlib
import json
import copy

class PreProcessor:
    def __init__(self, pp_name):
        self.pp_name = pp_name
        self.transformations = []
        self.pp_hash = '0'

        try:
            from secret_settings import LOG_LEVEL
            log_level = LOG_LEVEL
        except ImportError:
            log_level = 10  # DEBUG
        self.logger = get_console_logger(self.__class__.__name__, log_level)

    def add_transformation(self, transformation):
        assert isinstance(transformation, StatelessTransformation)
        self.transformations.append(transformation)
        self._update_hash()
        return self

    def clean_text(self, item):  # Explicitly required by scoring.py
        return self.preproc_one(item)

    def preproc_one(self, item):
        result = item
        for transformation in self.transformations:
            result = transformation.apply_one(result)
        return result

    def preproc_many(self, items):
        result = items
        for transformation in self.transformations:
            result = transformation.apply_many(result)
        return result

    def to_info_dict(self):
        pp_info = {
            'pp_name': self.pp_name,
            'pp_hash': self.pp_hash,
            'transformations': [repr(tr) for tr in self.transformations]
        }
        return pp_info

    def to_json(self, filename):
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(self.to_info_dict(), json_file, indent=2)

    @classmethod
    def from_info_dict(cls, info_dict):
        pp = cls(info_dict['pp_name'])
        pp.pp_hash = info_dict.get('pp_hash', '0')
        transformations_repr = info_dict.get('transformations', [])

        # ✅ Define safe globals so eval() can find required functions
        safe_globals = {
            'StatelessTransformation': StatelessTransformation,
            'lower': lower,
            'upper': upper,
            'replace': replace,
            'ljust': ljust,
            'lstrip': lstrip,
            'rstrip': rstrip,
            'strip': strip,
            'split': split
        }

        pp.transformations = [eval(tr_repr, safe_globals) for tr_repr in transformations_repr]
        return pp

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as json_file:
            info_dict = json.load(json_file)
        return cls.from_info_dict(info_dict)

    def _update_hash(self):
        pp_hash = hashlib.sha1()
        for transformation in self.transformations:
            pp_hash.update(repr(transformation).encode('utf-8'))
        self.pp_hash = pp_hash.hexdigest()
