from abc import ABC, abstractmethod
from datetime import datetime
from capture_model.utils import get_console_logger


class Model(ABC):

    datetime_format = '%Y-%m-%d %H:%M:%S'

    @abstractmethod
    def __init__(self, model_name):
        self.model_name = model_name
        self.init_datetime = datetime.now()
        self.class_name = self.__class__.__name__
        self.model_hash = None
        self.input_names = None
        self.output_names = None

        try:
            from secret_settings import LOG_LEVEL
            log_level = LOG_LEVEL
        except ImportError:
            log_level = 10  # DEBUG
        self.logger = get_console_logger(self.class_name, log_level)

    @abstractmethod
    def train(self, dataframe, input_names, output_names, *args, **kwargs):
        pass

    @abstractmethod
    def predict_one(self, item, *args, **kwargs):
        pass

    @abstractmethod
    def predict_many(self, items, *args, **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, items, *args, **kwargs):
        pass

    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def _to_model_info(self):
        model_info = {'model_name': self.model_name,
                      'init_datetime': self.init_datetime,
                      'class_name': self.class_name,
                      'model_hash': self.model_hash,
                      'input_names': self.input_names,
                      'output_names': self.output_names}

        return model_info

    def _handle_model_attr_value(self, attr_name, attr_value):
        if attr_name == 'init_datetime':
            return datetime.strptime(attr_value, self.datetime_format)
        else:
            return attr_value