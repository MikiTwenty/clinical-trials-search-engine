"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Logger class, designed for flexible and efficient logging of messages in Python applications.
The Logger class provides a streamlined way to log messages of different severities, such as informational, warning, error, and success messages. It supports optional logging to a file and can be configured to raise exceptions on error messages. The class is versatile for use in various parts of a Python project, aiding in debugging, monitoring, and reporting the application's operational status. Additionally, it supports color-coded console outputs for better readability and differentiation of log types.
"""

# Standard Library
import os
import datetime
from typing import Union

# Third-Party
from typeguard import typechecked


class COLORS:
    YELLOW = '\033[33m'
    RED = '\033[31m'
    WHITE = '\033[0m'
    GREEN = '\033[32m'
    BLUE = '\033[94m'

class Logger:

    @typechecked
    def __init__(
            self,
            id: str,
            log_file_path: Union[str, os.PathLike, None] = None,
            raise_errors: bool = True,
            verbose: Union[bool, int] = False
        )-> None:
        """
        Initialize a Logger instance.

        Args:
            id (str): Identifier for the logger.
            log_file_path (Union[str, os.PathLike, None]): File path for logging messages. None disables file logging. Defaults to None.
            raise_errors (bool): If True, raises errors on error logging. Defaults to True.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        self.log_file_path = log_file_path or os.getenv('LOG_FILE_PATH')
        self.id = id
        self.raise_errors = raise_errors
        self.verbose = verbose

    @typechecked
    def _save(
            self,
            text
        ) -> None:
        """
        Save the log message to a file.

        Args:
            text (str): The text to log.
        """

        if self.log_file_path and os.path.isdir(os.path.dirname(self.log_file_path)):
            log = f"[{datetime.datetime.now()}] {text}\n"
            with open(self.log_file_path, 'a') as file:
                file.write(log)

    @typechecked
    def warning(
            self,
            text: str,
        ) -> None:
        """
        Log a warning message.

        Args:
            text (str): The warning message to log.
        """

        _text = f"WARNING: [{self.id}] {text}"
        self._save(_text)

        if self.verbose:
            print(f"{COLORS.YELLOW}WARNING{COLORS.WHITE}: [{self.id}] {text}")

    @typechecked
    def error(
            self,
            text: str,
        ) -> None:
        """
        Log an error message and raise a RuntimeError if raise_errors is True.

        Args:
            text (str): The error message to log.
        """

        _text = f"ERROR: [{self.id}] {text}"
        self._save(_text)

        print(f"{COLORS.RED}ERROR{COLORS.WHITE}: [{self.id}] {text}")

        if self.raise_errors:
            raise RuntimeError(text)

    @typechecked
    def success(
            self,
            text: str,
        ) -> None:
        """
        Log a success message.

        Args:
            text (str): The success message to log.
        """

        _text = f"SUCCESS: [{self.id}] {text}!"
        self._save(_text)

        if self.verbose:
            print(f"{COLORS.GREEN}SUCCESS{COLORS.WHITE}: [{self.id}] {text}!")

    @typechecked
    def info(
            self,
            text: str,
        ) -> None:
        """
        Log an informational message.

        Args:
            text (str): The informational message to log.
        """

        _text = f"(i) [{self.id}] {text}"
        self._save(_text)

        if self.verbose:
            print(f"{COLORS.BLUE}(i){COLORS.WHITE} [{self.id}] {text}")


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")