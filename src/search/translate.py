"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Translator class, which uses Google Translate endpoints to perform text translations.
"""

# Standard Library
from json import loads
from typing import Optional
from requests import get, RequestException

# Third-Party
from typeguard import typechecked

# Local
from ..tools.logging import Logger


class ENDPOINTS:
    CHROME = "https://clients5.google.com/translate_a/t?client=dict-chrome-ex"
    GAPIS = "https://translate.googleapis.com/translate_a/single?client=gtx&dt=t"


class Translator:

    @typechecked
    def __init__(
            self,
            endpoint: str = ENDPOINTS.CHROME,
            verbose: bool = False
        ) -> None:

        """
        Initialize the Translator class.

        This class uses a specified Google Translate endpoint to perform translations.

        Args:
            endpoint (str): The URL of the Google Translate endpoint. Defaults to ENDPOINTS.CHROME.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        logger = Logger(
            id = 'Translator',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

        self.endpoint = endpoint

        self.info(f"Set endpoint: \"{self.endpoint}\"")

    @typechecked
    def translate(
            self,
            query: str,
            source_language: str = 'auto',
            target_language: str = 'en',
            raise_errors: bool = False
        ) -> Optional[str]:

        """
        Translate the given text from the source language to the target language.

        Args:
            query (str): The text to be translated.
            source_language (str): The language code of the source text. Default to 'auto' (automatic detection).
            target_language (str): The language code to translate the text into. Defaults to 'en'.
            raise_errors (bool): Raise error if there are network errors. Defaults to False.

        Returns:
            Optional[str]: The translated text, or None if an error occurred.
        """

        url = f"{self.endpoint}&sl={source_language}&tl={target_language}&q={query}"

        try:
            response = get(url)

            if raise_errors:
                response.raise_for_status()  # Raises HTTPError, if one occurred

            translated_text = loads(response.text)[0][0]

        except RequestException as e:
            self.warning(f"Error during translation request: {e}")
            return query

        self.info(f"Translated query: \"{translated_text}\"")

        return translated_text


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")