import collections
import datetime
import os
from pathlib import Path
import typing as t

from src.utils.downloads import check_download


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

        # Logs initialized
        self._initialized_loggers = collections.defaultdict(bool)

# ------------------ UTILS ------------------

    def remove_prefix(input_string: str, prefix: str) -> str:
        """Removes the prefix if exists at the beginning in the input string
        Needed for Python<3.9
        
        :param input_string: The input string
        :param prefix: The prefix
        :returns: The string without the prefix
        """
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def as_relative(self, path: t.Union[str, Path]) -> Path:
        """Removes the prefix `self.root` from an absolute path.

        :param path: The absolute path
        :returns: A relative path starting at `self.root`
        """
        if type(path) == str:
            path = Path(path)
        return Path(CustomizedPath.remove_prefix(path.as_posix(), self.root.as_posix()))

    def mkdir_if_not_exists(self, path: Path, gitignore: bool=False) -> Path:
        """Makes the directory if it does not exists

        :param path: The input path
        :param gitignore: A boolean indicating if a gitignore must be included for the content of the directory
        :returns: The same path
        """
        path.mkdir(parents=True, exist_ok = True)

        if gitignore:
            with (path / '.gitignore').open('w') as f:
                f.write('*\n!.gitignore')

        return path

# ------------------ MAIN FOLDERS ------------------

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.mkdir_if_not_exists(self.root / 'data', gitignore=True)

    @property
    def output(self):
        return self.mkdir_if_not_exists(self.root / 'output', gitignore=True)

    @property
    def logs(self):
        return self.mkdir_if_not_exists(self.root / 'logs', gitignore=True)

# ------------------ LOGS ------------------

    def get_log_file(self, logger_name: str) -> Path:
        """Creates and initializes a logger.

        :param logger_name: The logger name to create
        :returns: A path to the `logger_name.log` created and/or initialized file
        """
        file_name = logger_name + '.log'
        result = self.logs / file_name

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._initialized_loggers[logger_name]:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._initialized_loggers[logger_name] = True

        return result

# ------------------ DATA ------------------

    @property
    def train_data_path(self):
        return check_download(self.data / 'training_data.pkl')

    @property
    def test_data_path(self):
        return check_download(self.data / 'test_data.pkl')

    @property
    def labels_data_path(self):
        return check_download(self.data / 'training_labels.pkl')

# ------------------ KERNELS ------------------

    @property
    def kernels(self):
        return self.mkdir_if_not_exists(self.output / 'kernels')

    def get_kernel_file_path(self, kernel_name):
        return self.kernels / f'kernel_{kernel_name}.pkl'
    
# ------------------ PREDICTIONS ------------------

    @property
    def predictions(self):
        return self.mkdir_if_not_exists(self.output / 'predictions')

    def get_new_prediction_file(self, note: str = '') -> Path:
        """Get a new prediction file.

        :param note: A note that will be inserted in the name of the file.
        :returns: A Path to the file, without creating the file.
        """
        return self.predictions / f'prediction_{datetime.datetime.now().strftime("_%Y_%m%d__%H_%M_%S")}_{note}.csv'
    
    @property
    def best_prediction(self):
        return self.root / 'test_pred.csv'

project = CustomizedPath() 