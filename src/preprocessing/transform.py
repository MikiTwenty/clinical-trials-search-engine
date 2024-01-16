"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Transformer class, designed for text processing and analysis.
It provides methods for natural language processing and autocompletion.
"""

# Standard Library
import os
import re
import nltk
import spacy
import pickle
from typing import Union, List, Tuple, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-Party
from tqdm import tqdm
from nltk.corpus import words
from typeguard import typechecked

# Local
from ..tools.logging import Logger


class Transformer:

    @typechecked
    def __init__(
            self,
            save_path: Union[str, os.PathLike],
            verbose: bool = False
        ) -> None:
        """
        Initialize the Transformer class.

        Args:
            save_path (Union[str, os.PathLike]): Path to the file storing the processed files.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        self.save_path = save_path

        logger = Logger(
            id = 'Transformer',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

        self.info(f"spaCy v{spacy.__version__}")
        self.info(f"nltk v{nltk.__version__}")

        try:
            nltk.download('words', raise_on_error=True)
            self.english_dict = set(words.words())

        except Exception as e:
            self.error(f"Failed to download nltk files: {e}")

        self.success("nltk required files downloaded")

        try:
            self.nlp = spacy.load("en_core_web_sm")

        except Exception:
            try:
                self.info(f"Downloading spaCy files...")
                os.system("python -m spacy download en_core_web_sm --quiet")

            except Exception as e:
                self.error(f"Error loading spaCy model: {e}")

        self.success("spaCy initialized")

    def _clean_text(
            self,
            text: str
        ) -> str:
        """
        Clean the text by removing double spaces, carriage returns, and newline characters.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """

        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.replace("\r", ' ').replace("\n", ' ').replace("/", ' ').replace("\'", ' ').replace("*", ' ')

        return text.strip()

    @typechecked
    def _process_text(
            self,
            text: str
        ) -> str:
        """
        Process the given text by cleaning it and extracting lemmatized, non-stopword tokens.

        Args:
            text (str): The text to be processed.

        Returns:
            str: A string of processed text with lemmatized tokens.
        """

        text = self._clean_text(text)
        tokens = self.nlp(text)
        processed_tokens = set()

        for token in tokens:
            if not token.is_stop and not token.is_punct:
                processed_tokens.add(token.lower())
                processed_tokens.add(token.lemma_.lower())

        return ' '.join(processed_tokens)

    @typechecked
    def _process_doc(
            self,
            doc: Dict[str, str]
        ) -> Dict[str, str]:
        """
        Process a document represented as a dictionary, applying text processing to specific fields.

        Args:
            doc (Dict[str, str]): The document to process, represented as a dictionary.

        Returns:
            Dict[str, str]: A dictionary representing the processed document.
        """

        processed_doc = {
            'docno': doc.get('docno', ''),
            'text': self._process_text(doc.get('text', '')),
            'title': doc.get('title', ''),
            'summary': doc.get('summary', '')
        }

        return processed_doc

    @typechecked
    def _process_batch(
            self,
            batch: List[Tuple[Union[str, os.PathLike], str]],
            batch_index: int,
            total_batches: int
        ) -> List[Dict[str, Any]]:
        """
        Process a batch of files.

        Args:
            batch (List[Tuple[Union[str, os.PathLike], str]]): A list of tuples containing folder paths and filenames.
            batch_index (int): The current batch index.
            total_batches (int): The total number of batches.

        Returns:
            List[Dict[str, Any]]: A list of processed document dictionaries.
        """

        documents = []

        with ThreadPoolExecutor() as executor:
            future_to_doc = {executor.submit(self._process_doc, folder_path, filename): filename for folder_path, filename in batch}

            desc = f"Batch [{batch_index}/{total_batches}]"

            for future in tqdm(as_completed(future_to_doc), total=len(batch), desc=desc, leave=False):
                try:
                    document = future.result()
                    documents.append(document)

                except Exception as e:
                    self.error(f"Error processing file {future_to_doc[future]}: {e}")

        return documents

    @typechecked
    def _save(
            self,
            processed_documents: List[Dict[str, str]]
        ) -> None:
        """
        Save processed documents to a file.

        Args:
            processed_documents (List[Dict[str, Any]]): The list of processed documents.
        """

        save_file = os.path.join(self.save_path, "TREC21_processed.pkl")

        self.info(f"Saving documents to {save_file}")

        try:
            with open(save_file, 'wb') as f:
                pickle.dump(
                    obj = processed_documents,
                    file = f,
                    protocol = pickle.HIGHEST_PROTOCOL
                )

        except Exception as e:
            self.error(f"Failed saving documents to {save_file}: {e}")

        self.success("Processed documents saved as \"{save_file}\"")

    @typechecked
    def process_docs(
            self,
            documents: List[Dict[str, str]],
            parallel_processing: bool = True,
            batch_size: int = 100
        ) -> List[Dict[str, Any]]:
        """
        Process a list of documents either in parallel or sequentially.

        Args:
            documents (List[Dict[str, str]]): A list of document dictionaries.
            parallel_processing (bool): Flag to indicate whether to use parallel processing. Defaults to True.
            batch_size (int): The size of each batch for parallel processing. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: A list of processed document dictionaries.
        """

        processed_documents = []

        if parallel_processing:
            num_batches = (len(documents) + batch_size - 1) // batch_size

            with ThreadPoolExecutor() as executor:
                for batch_index in range(num_batches):
                    batch_start = batch_index * batch_size
                    batch_end = min(batch_start + batch_size, len(documents))
                    batch = documents[batch_start:batch_end]
                    futures = [executor.submit(self._process_doc, doc) for doc in batch]

                    for future in tqdm(as_completed(futures), total=len(batch), desc=f"Processing batch {batch_index + 1}/{num_batches}", leave=False):
                        try:
                            processed_doc = future.result()
                            processed_documents.append(processed_doc)
                        except Exception as e:
                            self.error(f"Error processing document: {e}")

        else:
            for doc in tqdm(documents, desc="Processing files sequentially"):
                processed_doc = self._process_doc(doc)
                processed_documents.append(processed_doc)

        self.success(f"Documents processed")

        self._save(processed_documents)

        return processed_documents

    @typechecked
    def process_query(
            self,
            query: str
        ) -> str:
        """
        Process a given query by tokenizing, removing stopwords and punctuation.

        Args:
            query (str): The query string to process.

        Returns:
            str: The processed query.
        """

        processed_query = self._process_text(query)

        self.info(f"Processed query: \"{processed_query}\"")

        return processed_query

    @typechecked
    def autocomplete(
            self,
            query: str
        ) -> List[str]:
        """
        Provides autocomplete suggestions for a given query based on a dictionary of English words.

        Args:
            query (str): The partial query string for which autocomplete suggestions are needed.

        Returns:
            List[str]: A list of suggested completions for the query.
        """

        try:
            words = query.split()
            if not words:
                return []

            last_word = words[-1]
            preceding_words = words[:-1]
            preceding_text = ' '.join(preceding_words) + ' ' if preceding_words else ''

            matches = [preceding_text + word for word in self.english_dict if word.startswith(last_word)]

            return matches

        except Exception as e:
            self.warning(f"Failed to load suggestions: {e}")
            return []


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")