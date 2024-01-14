"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the ClinicalBERT class which is designed for generating embeddings from clinical text using a pre-trained BERT model.
It provides methods for embedding generation and similarity matrix calculation.
"""

# Standard Library
import os

# Third-Party
import torch
from typing import Union, List, Dict
from transformers import AutoModel, AutoTokenizer
from typeguard import typechecked

# Local
from ..tools.logging import Logger


class ClinicalBERT:

    @typechecked
    def __init__(
            self,
            model_path: Union[str, os.PathLike],
            verbose: bool = False
        ) -> None:
        """
        Initialize ClinicalBERT with a specified model path.

        Args:
            model_path (Union[str, os.PathLike]): The file path or model identifier.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        logger = Logger(
            id = 'ClinicalBERT',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model: torch.nn.Module = AutoModel.from_pretrained(model_path)
            torch.cuda.empty_cache()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        except Exception as e:
            self.error(f"Failed loading model: {e}")

        self.success(f"Model initialized (device: {'GPU' if torch.cuda.is_available() else 'CPU'})")

    @typechecked
    def _get_embeddings(
            self,
            documents: List[str],
            max_length: int = 512
        ) -> torch.Tensor:
        """
        Generate embeddings for a list of documents using ClinicalBERT.

        Args:
            documents (List[str]): A list of documents (text) to process.
            max_length (int): Maximum number of tokens per document. Defaults to 512.

        Returns:
            torch.Tensor: Tensor containing embeddings for each document.
        """

        try:
            inputs = self.tokenizer(
                documents,
                padding = True,
                truncation = True,
                max_length = max_length,
                return_tensors = "pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

        except Exception as e:
            self.error(f"Failed to get Embeddings: {e}")

        self.success(f"Embeddings obtained")

        return embeddings

    @typechecked
    def _get_similarity_matrix(
            self,
            embeddings: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate the similarity matrix for a set of embeddings.

        Args:
            embeddings (torch.Tensor): A tensor containing embeddings for each document.

        Returns:
            torch.Tensor: A tensor representing the similarity matrix.
        """

        try:
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.transpose(0, 1))

        except Exception as e:
            self.error(f"Failed to get Similiraty Matrix: {e}")

        self.info(f"Similarity Matrix:\n{similarity_matrix}\n")

        return similarity_matrix

    @typechecked
    def __call__(
            self,
            documents: List[Dict],
            ids: List[str]
        ) -> torch.Tensor:
        """
        Generate embeddings for a list of documents.

        Args:
            documents (List[str]): A list of documents.
            ids (List[str]): A list of document IDs to be processed.

        Returns:
            torch.Tensor: Tensor containing embeddings.
        """

        documents_text = [document['text'] for document in documents if document['docno'] in ids]

        if not documents_text:
            self.error("No matching documents found for the provided IDs")

        embeddings = self._get_embeddings(documents_text)

        similarity_matrix = self._get_similarity_matrix(embeddings)

        return similarity_matrix


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")