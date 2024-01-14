"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Pipeline class, which orchestrates various components of the CTSE project such as data transformation, indexing, translation, expansion, ranking, and storage.
It manages the flow of data through these components and provides methods for retrieving, ranking, and re-ranking documents, as well as storing user interactions.
"""

# Standard Library
import os
import pickle
from typing import Any, List, Dict, Optional

# Third-Party
from typeguard import typechecked

# Local
from .tools.logging import Logger
from .search.llm import LLM
from .evaluation.ranking import Ranker
from .search.translate import Translator
from .evaluation.embed import ClinicalBERT
from .tools.utils import get_topn, open_xml
from .preprocessing.indexing import Indexer
from .preprocessing.transform import Transformer


class Pipeline:

    @typechecked
    def __init__(
            self,
            paths: Dict[str, str],
            verbose: bool = False
        ) -> None:
        """
        Initialize the Pipeline class.

        Args:
            paths (Dict[str, str]): The paths dictionary.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        self.paths =  paths
        self.verbose = verbose

        logger = Logger(
            id = 'Pipeline',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

        self.topn = int
        self.original_query: str = ''
        self.translated_query: str = ''
        self.expanded_query: str = ''
        self.original_results: list = []
        self.translated_results: list = []
        self.expanded_results: list = []
        self.original_top: list = []
        self.translated_top: list = []
        self.expanded_top: list = []
        self.similarity_matrix: list = []
        self.ranked_list: list = []
        self.reranked_list: list = []

        self.transformer = Transformer(
            save_path = self.paths['DATA'],
            verbose = self.verbose
        )

        self.indexer = Indexer(
            jdk_path = self.paths['JDK'],
            file_dir = self.paths['INDEXING_FILES'],
            verbose = self.verbose
        )

        self.translator = Translator()

        self.llm = LLM(
            model_path = self.paths['LLM'],
            verbose = self.verbose
        )

        self.cBERT = ClinicalBERT(
            model_path = self.paths['BERT'],
            verbose = self.verbose
        )

        self.ranker = Ranker(
            verbose = self.verbose
        )

        self._load_documents()

    def _load_documents(
            self
        ) -> None:
        """
        Load documents from a file.
        """

        try:
            with open(self.paths['DATASET'], 'rb') as file:
                self.documents = pickle.load(file)

        except Exception as e:
            self.error(f"Failed to load documents from {self.paths['DATASET']}: {e}")

        self.success(f"Documents loaded")

    @typechecked
    def retrieve_original(
            self,
            query: str,
            topn: int = 10
        ) -> None:
        """
        Retrieve original documents based on the query.

        Args:
            query (str): The query to retrieve documents for.
            topn (int): Number of top documents to retrieve. Defaults to 10.
        """

        self.info(f"Original Documents Retrieval...")

        self.topn = topn

        self.original_query = self.transformer.process_query(
            query = query
        )

        self.original_results = self.indexer.retrieve(
            query = self.original_query,
            baseline = 'TF_IDF',
            expansion = None
        )

        self.original_top = get_topn(
            results = self.original_results,
            n = topn,
            verbose = self.verbose
        )

    def retrieve_translated(
            self
        ):
        """
        Retrieve documents based on the translated query.
        """

        self.info(f"Translated Documents Retrieval...")

        self.translated_query = self.translator.translate(
            query = self.original_query
        )

        self.translated_query = self.transformer.process_query(
            query = self.translated_query
        )

        self.translated_results = self.indexer.retrieve(
            query = self.translated_query,
            baseline = 'TF_IDF',
            expansion = 'Bo1'
        )

        self.translated_top = get_topn(
            results = self.translated_results,
            n = self.topn,
            verbose = self.verbose
        )

    def retrieve_expanded(
            self
        ) -> str:
        """
        Retrieve documents based on the expanded query.

        Returns:
            str: The medical condition.
        """

        self.info(f"Expanded Documents Retrieval...")

        self.expanded_query, medical_condition = self.llm.expand_query(
            query = self.translated_query
        )

        self.expanded_query = self.transformer.process_query(
            query = self.expanded_query
        )

        self.expanded_results = self.indexer.retrieve(
            query = self.expanded_query,
            baseline = 'TF_IDF',
            expansion = None
        )

        self.expanded_top = get_topn(
            results = self.expanded_results,
            n = self.topn,
            verbose = self.verbose
        )

        return medical_condition

    @typechecked
    def _evaluate(
            self,
            ids: Optional[List[str]] = None
        ) -> None:
        """
        Evaluate document similarity.

        Args:
            ids (Optional[List[str]]): List of document ids to include in the similarity evaluation. Defaults to None.
        """

        if not ids:
            combined_top_ids_set = set()
            for top_list in [self.original_top, self.translated_top, self.expanded_top]:
                for doc_id, _ in top_list:
                    combined_top_ids_set.add(doc_id)
            ids = list(combined_top_ids_set)

        self.similarity_matrix = self.cBERT(
            ids = ids,
            documents = self.documents
        )

    def rank(
            self
        ) -> List[Dict[str, Any]]:
        """
        Rank documents based on similarity and relevance.

        Returns:
            List[Dict[str, Any]]: A list of ranked documents.
        """

        self.info(f"Documents Ranking...")

        topn = self.topn*3 if self.expanded_top else self.topn*2

        self.ranked_list = self.ranker.rank(
            self.original_top,
            self.translated_top,
            self.expanded_top,
            topn = topn,
            nmax = topn*3
        )

        return self.ranked_list

    @typechecked
    def rerank(
            self,
            ranked_list: List[Dict[str, Any]],
            feedback: Dict[str, Any]
        ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on user feedback.

        Args:
            ranked_list (List[Dict[str, Any]]): The initial ranked list of documents.
            feedback (Dict[str, Any]): User feedback for reranking.

        Returns:
            List[Dict[str, Any]]: The re-ranked list of documents.
        """

        ids = set()
        for item in ranked_list:
            ids.add(item['docno'])

        self._evaluate(ids=list(ids))

        self.info(f"Documents Reranking...")

        self.reranked_list = self.ranker.rerank(
            ranked_list = ranked_list,
            feedback = feedback,
            similarity_matrix = self.similarity_matrix
        )

        return self.reranked_list

    @typechecked
    def open_file(
            self,
            docno: str
        ) -> None:
        """
        Open an XML file associated with a given document number.

        Args:
            docno (str): The document number of the file to open.
        """

        docno = f"{docno}.xml"

        self.info(f"Opening file: {docno}")

        for root, dirs, files in os.walk(self.paths['DOCUMENTS']):
            for file in files:
                if file == docno:
                    xml_path = os.path.join(root, file)

        if not xml_path:
            self.warning(f"File not found: {docno}")
            return

        open_xml(
            path = xml_path
        )


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")