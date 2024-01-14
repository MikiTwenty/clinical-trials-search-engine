"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module provides a Python wrapper for LLMs, facilitating query expansion and other text processing tasks.
It is designed to work with the llama-cpp library.
"""

# Standard Library
import os
import re
from typing import Any, Union

# Third-Party
from llama_cpp import Llama as Model, __version__
from torch.cuda import is_available as gpu
from typeguard import typechecked

# Local
from ..tools.logging import Logger


class LLM:

    @typechecked
    def __init__(
            self,
            model_path: Union[str, os.PathLike],
            verbose: bool = False
        ) -> None:
        """
        Initialize the Large Lenguage Model wrapper.

        Args:
            model_path (Union[str, os.PathLike]): Path to the Model GGUF file.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        base_name = os.path.basename(model_path)
        self.model_name = base_name.split('.')[0].capitalize().replace('-', ' ').split()[0]

        self.logger = Logger(
            id = self.model_name,
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = self.logger.warning
        self.error = self.logger.error
        self.info = self.logger.info
        self.success = self.logger.success

        self.info(f"llama-cpp-python v{__version__}")

        try:
            self.model = Model(
                model_path = model_path,
                n_gpu_layers = 20 if gpu() else 0,  # The number of offloaded layers can vary basing on your gpu
                n_threads = os.cpu_count() or None,
                verbose = verbose  # Very long output!
            )

            self._warmup()

            self.success(f"Model initialized (device: {'GPU' if gpu() else 'CPU'})")

        except Exception as e:
            self.error(f"Model initialization failed: {e}")

    def _warmup(
            self
        ) -> None:
        """
        Warmup the model after the initialization.
        """

        # Run an empty query expansion with hidden output
        _v = self.logger.verbose
        self.logger.verbose = False
        self.expand_query(" ")
        self.logger.verbose = _v

    @typechecked
    def expand_query(
            self,
            query: str,
            max_tokens: int = 32,
            temp: float = 0.1
        ) -> Any:
        """
        Expand a given query with the extracted medical condition.

        Args:
            query (str): The query to be expanded.
            max_tokens (int): The maximum number of new tokens. Defaults to 32.
            temp (float): The temperature parameter for the model. Defaults to 0.1.

        Returns:
            str: The expanded query.
        """

        query_dict = set(word for word in query.split())

        instructions = (
            "Your must extract a medical condition from the given prompt. "
            "If a medical condition is found, your answer must be just the medical condition. "
            "You should not answer if no medical condition is found! "
            "Your answer must not include comments or opinions!"
        )

        prompt = f"<s>[INST] <<SYS>>\n{instructions}\n<</SYS>>\n\n{query} [/INST]"

        max_tokens = len(self.model.tokenize(query.encode('utf-8'))) + max_tokens

        output = self.model(
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = temp,
            echo = False
        )

        if not output:
            self.warning(f"Failed to extract medical condition and expand the query, return the original query!")
            return query

        text_output = self._clean_output(output["choices"][0]["text"])
        query_dict.update(word for word in text_output.split())
        expanded_query = ' '.join(query_dict)

        self.info(f"Medical condition: \"{text_output}\"")
        self.info(f"Expanded query: \"{expanded_query}\"")

        return expanded_query, text_output

    @typechecked
    def _clean_output(
            self,
            text: str
        ) -> str:
        """
        Clean the model output by removing letter markers and numbers.

        Args:
            text (str): The text output from the model.

        Returns:
            str: The cleaned text.
        """

        return re.sub(r'\b[A-Z0-9]+:\s?', '', text)

    # __enter__ and __exit__ needed for evaluation loop in 'examples/eval.py'
    def __enter__(
            self
        ) -> Any:
        """
        Enter the runtime context related to this object.

        The with statement will bind this method's return value
        to the target specified in the as clause of the statement, if any.
        """
        return self

    def __exit__(
            self,
            exc_type: Any,
            exc_val: Any,
            exc_tb: Any
        ) -> False:
        """
        Exit the runtime context and perform any cleanup actions.

        Args:
            exc_type: Exception type if raised in context.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.

        Returns:
            False
        """

        if exc_type:
            self.error(f"An error occurred: {exc_val}")

        return False



if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")