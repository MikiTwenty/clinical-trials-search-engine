"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Ranker class which provides functionality for ranking, normalizing, and reranking documents based on various criteria.
It includes methods for normalizing scores, removing duplicates, and generating ranked lists of documents.
"""

# Standard Library
from typing import List, Dict, Tuple, Union

# Third-Party
from torch import Tensor
from typeguard import typechecked

# Local
from ..tools.logging import Logger


class Ranker:
    """
    The Ranker class provides methods to rank documents based on their scores,
    remove duplicate documents, and re-rank them based on user feedback and similarity matrices.
    """

    @typechecked
    def __init__(
            self,
            verbose: bool = False
        ) -> None:
        """
        Initializes the Ranker class with empty lists for original, translated, and expanded documents.

        Args:
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
        """

        self.verbose = verbose
        self.original_docs = []
        self.expanded_docs = []

        logger = Logger(
            id = 'Ranker',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

    def _normalize(
            self
        ) -> None:
        """
        Normalizes the scores of documents in original, translated, and expanded lists to a 0-1 range.
        """

        try:
            for doc_list in [self.original_docs, self.translated_docs, self.expanded_docs]:
                if not doc_list:
                    continue

                max_score = 1
                min_score = 0

                if max_score == min_score:
                    continue

                for doc in doc_list:
                    doc['score'] = (doc['score'] - min_score) / (max_score - min_score)

        except Exception as e:
            self.error(f"Failed to normalize the scores: {e}")

        self.success("Scores normalized")

    def _remove_doubles(
            self
        ) -> None:
        """
        Removes duplicate documents from the translated and expanded lists based on document numbers.
        """

        try:
            original_docnos = {doc['docno'] for doc in self.original_docs}
            self.translated_docs = [doc for doc in self.translated_docs if doc['docno'] not in original_docnos]
            self.expanded_docs = [doc for doc in self.expanded_docs if doc['docno'] not in original_docnos]

        except Exception as e:
            self.error(f"Failed to remove duplicates from retrieved documents: {e}")

        self.success(f"Retrieved documents duplicates removed")

    @typechecked
    def _get_ranked_list(
            self,
            topn: int,
            nmax: int
        ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Generates a ranked list of documents combining original, translated, and expanded documents.

        Args:
            topn (int): The number of top documents to consider from each category.
            nmax (int): The maximum number of documents to include in the final ranked list.

        Returns:
            List[Dict[str, Union[int, str, float]]]: A list of dictionaries representing ranked documents.
        """

        try:
            top_original_docs = sorted(
                self.original_docs,
                key = lambda x: x['score'],
                reverse = True
            )[:topn]

            top_translated_docs = sorted(
                self.translated_docs,
                key = lambda x: x['score'],
                reverse = True
            )[:topn]

            top_expanded_docs = sorted(
                self.expanded_docs,
                key = lambda x: x['score'],
                reverse = True
            )[:topn]

            remaining_original_docs = self.original_docs[topn:]
            remaining_translated_docs = self.original_docs[topn:]
            remaining_expanded_docs = self.expanded_docs[topn:]

            mixed_remaining_docs = remaining_original_docs + remaining_translated_docs + remaining_expanded_docs

            sorted_mixed_remaining_docs = sorted(
                mixed_remaining_docs,
                key = lambda x: x['score'],
                reverse = True
            )

            final_ranked_list = top_original_docs + top_translated_docs + top_expanded_docs + sorted_mixed_remaining_docs

            ranked_list = [
                {
                    'rank': rank + 1,
                    'docno': doc['docno'],
                    'score': doc['score'],
                    'from_query': doc['from_query']
                }
                for rank, doc in enumerate(final_ranked_list[:nmax])
            ]

        except Exception as e:
            self.error(f"Failed to get ranked list: {e}")

        if self.verbose:
            _results: str = ''
            for doc in ranked_list:
                _results += (f"\nRank: {doc['rank']} - ID: {doc['docno']}")
            self.info(f"Ranked documents:{_results}\n")

        return ranked_list

    @typechecked
    def rank(
            self,
            original_docs: List[Tuple[str, float]],
            translated_docs: List[Tuple[str, float]],
            expanded_docs: List[Tuple[str, float]],
            topn: int = 5,
            nmax: int = 10
        ) -> List[Dict]:
        """
        Ranks the documents based on provided lists of original, translated, and expanded documents.

        Args:
            original_docs (List[Tuple[str, float]]): List of tuples representing original documents.
            translated_docs (List[Tuple[str, float]]): List of tuples representing translated documents.
            expanded_docs (List[Tuple[str, float]]): List of tuples representing expanded documents.
            topn (int): The number of top documents to consider from each category. Defaults to 5.
            nmax (int): The maximum number of documents to include in the final ranked list. Defaults to 10.

        Returns:
            List[Dict]: A list of dictionaries representing the ranked documents.
        """

        self.info(f"Ranking {topn} topn documents...")

        try:
            self.original_docs = [{
                'docno': doc[0],
                'score': float(doc[1]),
                'from_query': 'original'
            } for doc in original_docs]

            self.translated_docs = [{
                'docno': doc[0],
                'score': float(doc[1]),
                'from_query': 'translated'
            } for doc in translated_docs]

            self.expanded_docs = [{
                'docno': doc[0],
                'score': float(doc[1]),
                'from_query': 'expanded'
            } for doc in expanded_docs]

        except Exception as e:
            self.error(f"Failed to transform retrieved documents: {e}")

        self.success(f"Retrieved documents transformed")

        self._normalize()
        self._remove_doubles()
        results = self._get_ranked_list(topn, nmax)

        return results

    @typechecked
    def rerank(
            self,
            ranked_list: List[Dict[str, Union[int, str, float]]],
            feedback: Dict[str, str],
            similarity_matrix: Tensor
        ) -> List[Dict[str, Union[int, str, float]]]:
        """
        Rerank the documents based on user feedback and a similarity matrix.

        Args:
            ranked_list (List[Dict[str, Union[int, str, float]]]): The initial ranked list of documents.
            feedback (Dict[str, str]): User feedback for a specific document.
            similarity_matrix (Tensor): A tensor representing the similarity between documents.

        Returns:
            List[Dict[str, Union[int, str, float]]]: The re-ranked list of documents.
        """

        try:
            selected_doc_rank = next((doc['rank'] for doc in ranked_list if doc['docno'] == feedback['docno']), None)
            if selected_doc_rank is None:
                return ranked_list

            selected_doc_index = selected_doc_rank - 1

            # Rerank all the documents: DISABLED
            # >>> for doc_to_update, current_doc in enumerate(ranked_list):
            # >>>    if doc_to_update == selected_doc_index:  # Skip the selected document
            # >>>        continue

            for doc_to_update in range(selected_doc_index + 1, len(ranked_list) - 1):
                current_doc = ranked_list[doc_to_update]
                sim_value = similarity_matrix[selected_doc_index, doc_to_update].item()

                if feedback['feedback'] == '+':
                    new_score = current_doc['score'] * sim_value

                elif feedback['feedback'] == '-':
                    similarity_list = similarity_matrix[selected_doc_index].tolist()
                    similarity_value = similarity_list[doc_to_update]

                    ordered_similarity = sorted(
                        similarity_list,
                        reverse = True
                    )

                    similarity_position = ordered_similarity.index(similarity_value)
                    inverse_sim_value = ordered_similarity[-similarity_position]
                    new_score = current_doc['score'] * inverse_sim_value

                else:
                    continue

                current_doc['score'] = new_score

            reranked_part = sorted(ranked_list[selected_doc_index + 1:], key=lambda x: x['score'], reverse=True)

            ranked_list = ranked_list[:selected_doc_index + 1] + reranked_part

            for rank, doc in enumerate(ranked_list[selected_doc_index + 1:], start=selected_doc_index + 2):
                doc['rank'] = rank

        except Exception as e:
            self.error(f"Failed to get reranked list: {e}")

        if self.verbose:
                _results: str = ''
                for doc in ranked_list:
                    _results += (f"\nRank: {doc['rank']} - ID: {doc['docno']}")
                self.info(f"Reranked documents:{_results}\n")

        return ranked_list


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")