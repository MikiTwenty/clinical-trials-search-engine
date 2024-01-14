"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Indexer class, which is responsible for initializing the PyTerrier framework,
indexing data, saving and loading retrievers, and performing document retrieval based on queries.
"""

# Standard Library
import os
import shutil
from typing import List, Dict, Any, Union

# Third-Party
import pandas as pd
import pyterrier as pt
from typeguard import typechecked

# Local
from ..tools.logging import Logger

class Indexer:

    @typechecked
    def __init__(
            self,
            jdk_path: Union[str, os.PathLike],
            file_dir: Union[str, os.PathLike],
            verbose: bool = False
        ) -> None:
        """
        Initialize the Indexer class.

        Args:
            jdk_path (Union[str, os.PathLike]): Path to the JDK required by PyTerrier.
            file_dir (Union[str, os.PathLike]): Directory to save files related to the indexer.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        logger = Logger(
            id = 'Indexer',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

        self.info(f"Pandas v{pd.__version__}")
        self.info(f"PyTerrier v{pt.__version__}")

        if not pt.started():
            try:
                os.environ["JAVA_HOME"] = jdk_path
                pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

            except Exception as e:
                self.error(f"Failed to initialize PyTerrier: {e}")

            self.success("PyTerrier initialized")

        self.file_dir = file_dir

    @typechecked
    def _clear_index_folder(
            self,
            folder_path: os.PathLike
        ) -> None:
        """
        Clears the specified index folder by deleting its contents.

        Args:
            folder_path (Union[str, os.PathLike]): The path to the index folder.
        """

        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                except Exception as e:
                    self.error(f"Failed to delete {file_path}. Reason: {e}")

                self.success("Index folder cleared!")

    @typechecked
    def index(
            self,
            data: List[Dict[str, Any]],
            index_folder: Union[str, os.PathLike]
        ) -> pt.IndexRef:
        """
        Indexes the given data and creates a retriever.

        Args:
            data (List[Dict[str, Any]]): The data to index. It should be a list of dictionaries containing 'text' and 'docno'.
            index_folder (Union[str, os.PathLike]): The path to the folder where the index will be stored.
            wmodel (str): The weighting model to use for indexing. Default is BM25.

        Returns:
            pt.IndexRef: A reference to the created index.
        """

        documents_df = pd.DataFrame(data)

        self._clear_index_folder(index_folder)

        self.indexer = pt.DFIndexer(
            index_path = index_folder,
            overwrite = True,
            verbose = True
        )

        try:
            self.index_ref = self.indexer.index(documents_df['text'], documents_df['docno'])

        except Exception as e:
            self.error(f"Indexing failed: {e}")

        self.success("Indexing complete")

        self.bm25 = pt.BatchRetrieve(
            self.index_ref,
            wmodel = 'BM25',
            controls = {
                "c": 1.0,
                "bm25.k1": 1.2,
                "bm25.b": 0.75
            }
        )

        self.tf_idf = pt.BatchRetrieve(
            self.index_ref,
            wmodel = 'TF_IDF'
        )

    def load(
            self
        ) -> None:
        """
        Loads the retriever state from a file.
        """

        try:
            self.index_ref = pt.IndexFactory.of(self.file_dir)
            self.bm25 = pt.BatchRetrieve(
                self.index_ref,
                wmodel = 'BM25',
                controls = {
                    "c": 1.0,
                    "bm25.k1": 1.2,
                    "bm25.b": 0.75
                }
            )

            self.tf_idf = pt.BatchRetrieve(
                self.index_ref,
                wmodel = 'TF_IDF'
            )

        except Exception as e:
            self.error(f"Fail to load indexing files: {e}")

        self.success("Indexing files loaded")

    @typechecked
    def retrieve(
            self,
            query: str,
            baseline: Union[str, Any] = 'TF_IDF',  # Any option to avoid conflict with typeguard.typedcheck
            expansion: Union[str, None] = None
        ) -> pd.DataFrame:
        """
        Retrieves documents based on the given query.

        Args:
            query (Union[str, os.PathLike]): The query string to search for.
            baseline (str): The baseline model to use for expansion. Choose between 'BM25' and 'TF_IDF'. Defaults to 'TF_IDF'
            expansion (Union[str, None]): The type of expansion. Choose between None, 'RM3', 'Bo1', 'KL', or 'Ax' (Axiomatic). Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the retrieval results.
        """

        baseline = baseline.lower()
        _baseline = self.bm25 if baseline == 'bm25' else self.tf_idf

        pipeline = _baseline

        # Best pipeline based on previous evaluation:
        if expansion:
            pipeline = self.expand_query(
                baseline = baseline,
                expansion = 'Bo1',
                nterms = int(len(query.split())/2),
                ndocs = 10
            )

        try:
            results = pipeline.search(query)

        except Exception as e:
            self.error(f"Search failed: {e}")

        self.info(f"Documents retrieved:\n{results}\n")

        return results

    @typechecked
    def expand_query(
            self,
            baseline: Union[str, Any] = 'TF_IDF',  # Any option to avoid conflict with typeguard.typedcheck
            expansion: str = 'RM3',
            nterms: int = 10,
            ndocs: int = 3
        ) -> Any:
        """
        Expand query using PyTerrier built-in methods.

        Args:
            baseline (str): The baseline retrieval system to use for expansion.
            expansion (str): The type of expansion. Choose between 'RM3', 'Bo1', 'KL', or 'Ax' (Axiomatic). Defaults to 'RM3'.
            nterms (int): Number of terms for feedback. Defaults to 10.
            ndocs (int): Number of documents for feedback. Defaults to 3.

        Returns:
            PyTerrier Retrieval System pipeline with query expansion applied.
        """

        _index_ref = self.index_ref

        query_expansion_methods = {
            'ax': pt.rewrite.AxiomaticQE(_index_ref, fb_terms=nterms, fb_docs=ndocs),
            'kl': pt.rewrite.KLQueryExpansion(_index_ref, fb_terms=nterms, fb_docs=ndocs),
            'bo1': pt.rewrite.Bo1QueryExpansion(_index_ref, fb_terms=nterms, fb_docs=ndocs),
            'rm3': pt.rewrite.RM3(_index_ref, fb_terms=nterms, fb_docs=ndocs)
        }

        if baseline.lower() == 'tf_idf':
            baseline = self.tf_idf
        elif baseline.lower() == 'bm25':
            baseline = self.bm25
        else:
            raise self.error(f"Invalid baseline '{expansion}'. Choose between 'TF_IDF' and 'BM25'.")

        if expansion.lower() in query_expansion_methods:
            return baseline >> query_expansion_methods[expansion.lower()] >> baseline

        else:
            raise self.error(f"Invalid query expansion method '{expansion}'. Choose between 'RM3', 'Bo1', 'KL', or 'Ax'.")


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")