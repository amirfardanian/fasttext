import copy
import pandas as pd


class StatelessTransformation:
    function_types = ['function']

    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = kwargs

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.function == other.function \
                   and self.kwargs == other.kwargs
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        string_list = []

        for key, value in sorted(self.to_dict().items()):
            if key == 'kwargs':
                kwargs_list = [str(k) + '=' + self._kwargs_value_repr(v) for k, v in sorted(value.items())]
                string_list.extend(kwargs_list)
            else:
                string_list.append(str(key) + '=' + str(value))

        representation = self.__class__.__name__ + '(' + ', '.join(string_list) + ')'

        return representation

    def _kwargs_value_repr(self, value):
        if type(value) == str:
            return "'{}'".format(value)
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            raise ValueError('__repr__() of pandas types is not supported. They are too big')
        else:
            return str(value)

    def apply_one(self, item):
        item_copy = copy.deepcopy(item)
        return self.function(item_copy, **self.kwargs)

    def apply_many(self, items):
        if isinstance(items, pd.Series):
            return self._apply_many_series(items)
        else:
            return self._apply_many_list(items)

    def _apply_many_list(self, items):
        result = []

        for item in items:
            applied = self.apply_one(item)
            result.append(applied)

        return result

    def _apply_many_series(self, items):
        return self.function(items, **self.kwargs)

    def to_dict(self):
        dct = {}
        for k, v in self.__dict__.items():
            if k in self.function_types:
                dct[k] = v.__name__
            else:
                dct[k] = v
        return dct
