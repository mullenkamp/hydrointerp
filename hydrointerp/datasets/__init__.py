import os


__all__ = ['available', 'get_path']

_module_path = os.path.dirname(__file__)
_available_data = {p.split('.')[0]: p for p in os.listdir(_module_path) if p.endswith('.nc')}
available = list(_available_data.keys())


def get_path(dataset):
    """
    Get the path to the data file.

    Parameters
    ----------
    dataset : str
        The name of the dataset. See ``nzmetservice.datasets.available`` for
        all options.

    """
    if dataset in _available_data:
        return os.path.abspath(
            os.path.join(_module_path, _available_data[dataset]))
    else:
        msg = "The dataset '{data}' is not available".format(data=dataset)
        raise ValueError(msg)
