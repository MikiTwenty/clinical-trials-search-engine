"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the GUI class, which provides a graphical user interface for the CTSE project.
It facilitates interaction with the underlying Pipeline for processing and evaluating search queries.
"""

# Standard Library
import os
from typing import Union, List, Dict, Optional

# Third-Party
from nicegui import ui
from typeguard import typechecked

# Local
from .pipeline import Pipeline
from .tools.logging import Logger



class GUI:
    """
    Graphical User Interface powered by NiceGUI.
    """

    @typechecked
    def __init__(
            self,
            paths: Optional[Dict[str, str]] = None,
            verbose: bool = False,
            _dev: bool = False
        ) -> None:
        """
        Initialize the Graphical User Interface (GUI).

        Args:
            paths (Optional[Dict[str, str]]): A dictionary of paths if already available.
            verbose (bool): If True, additional details are printed during operations. Defaults to False.
            _dev (bool): If True, more output.
        """

        self.verbose = verbose
        self._dev = _dev

        try:
            if paths['LOG']:
                os.environ['LOG_FILE_PATH'] = paths['LOG']
        except:
            pass

        logger = Logger(
            id = 'GUI',
            log_file_path = None,
            raise_errors = True,
            verbose = verbose
        )

        self.warning = logger.warning
        self.error = logger.error
        self.info = logger.info
        self.success = logger.success

        self.pipeline = Pipeline(
            paths = paths,
            verbose = self.verbose
        )

        self.search_history = []
        self.expansion = False

    @typechecked
    def _rank(
            self,
            query: str
        ) -> None:
        """
        Perform the initial ranking of documents based on the given query.

        Args:
            query (str): The query for document retrieval.
        """

        self.pipeline.indexer.load()

        try:
            self.pipeline.retrieve_original(query, topn=15)
            self.pipeline.retrieve_translated()
            if self.expansion:
                medical_condition = self.pipeline.retrieve_expanded()
                self.med_cond_label.text = f"Medical Condition: {medical_condition}"
            ranked_list = self.pipeline.rank()

        except Exception as e:
            self.error(f"Error during initial search: {e}")

        self.search_history.append({'query': query, 'results': ranked_list, 'feedback': {}})

    @typechecked
    def _rerank(
            self,
            feedback: dict
        ) -> None:
        """
        Re-rank documents based on user feedback.

        Args:
            feedback (dict): The feedback data for re-ranking.
        """

        last_state = self.search_history[-1]

        try:
            reranked_list = self.pipeline.rerank(
                ranked_list = last_state['results'],
                feedback = feedback
            )
            self.search_history.append({'query': last_state['query'], 'results': reranked_list, 'feedback': feedback})

        except Exception as e:
            self.error(f"Error during reranking: {e}")

        if self._dev and self.verbose:
            _text: str = ''
            for doc in self.search_history:
                _results_text = '\n > '.join([f"Rank: {result['rank']} - ID: {result['docno']} - Score: {result['score']} - From Query: {result['from_query']}" for result in doc['results']])
                _text += f"\n---\nQuery: {doc['query']}\nResults:\n > {_results_text}\nFeedback: {doc.get('feedback', 'None')}"
            self.info(f"Search history:{_text}\n")

    @typechecked
    async def search(
            self,
            query: str,
            reset: bool
        ) -> list:
        """
        Search for documents based on the given query.

        Args:
            query (str): The search query string.
            reset (bool): Indicates whether to reset the search history.

        Returns:
            dict: The search results.
        """

        query = query.strip()
        if not query:
            return {'results': []}

        if reset:
            self.search_history = []
            self._rank(query)

        latest_results = self.search_history[-1]['results'] if self.search_history else []

        results = self._search(latest_results)

        return results

    @typechecked
    def _search(
            self,
            latest_results: List[Dict]
        ) -> list:
        """
        Helper method to process the latest search results.

        Args:
            latest_results (dict): The latest search results to process.

        Returns:
            list: Processed search results.
        """

        docno_to_document = {document['docno']: document for document in self.pipeline.documents}

        return [{
            'docno': document.get('docno'),
            'title': document.get('title', 'No Title'),
            'summary': document.get('summary', 'No Summary')[:300] + ' ...'
        } for result in latest_results if (document := docno_to_document.get(result['docno']))]


    @typechecked
    async def feedback(
            self,
            record: dict,
            feedback_value: Union[int, None]
        ) -> None:
        """
        Process user feedback for a specific record.

        Args:
            record: The record to which the feedback is related.
            feedback_value (int): The value of the feedback.
        """

        if self.search_history:
            self._feedback(record, feedback_value)
            await self.display(
                query = self.search_field.value,
                reset_search = False
            )

    @typechecked
    def _feedback(
            self,
            record: dict,
            feedback_value: Union[int, None]
        ) -> None:
        """
        Internal method to handle feedback processing.

        Args:
            record (dict): The record for which the feedback is being processed.
            feedback_value (int): The value of the feedback.
        """

        truncate_index = None
        for i, val in enumerate(self.search_history):
            if val['feedback'] and val['feedback']['docno'] == record['docno']:
                truncate_index = i
                break

        if truncate_index is not None:
            self.search_history = self.search_history[:truncate_index]

        if feedback_value in [1, 2]:
            feedback_type = '-' if feedback_value == 1 else '+'
            self._rerank({'docno': record['docno'], 'feedback': feedback_type})

    @typechecked
    async def display(
            self,
            query: str,
            reset_search: bool = True
        ) -> None:
        """
        Displays search results for the given query and sets up feedback toggles for each result.

        Args:
            query (str): The search query string.
            reset_search (bool): Indicates whether to reset the search history.
        """

        query = query.strip()
        if not query:
            self.results.clear()
            return

        response = await self.search(
            query,
            reset = reset_search
        )

        self.results.clear()

        with self.results \
            .classes(
                'flex '
                'flex-col '
                'items-center '
                'justify-center '
                'w-full '
            ):

            for record in response:
                with ui.card() \
                    .classes(
                        'mb-4 '
                        'p-4 '
                        'w-3/4 '
                    ):

                    ui.label(record['title']) \
                        .classes(
                            'text-h6 '
                        ) \
                        .style(
                            'cursor: pointer;'
                        ) \
                        .on(
                            type = 'click',
                            handler = lambda e, docno = record['docno']:
                                self.pipeline.open_file(
                                    docno
                                )
                        )

                    ui.markdown(record['summary']) \
                        .classes(
                            'text-sm '
                            'overflow-hidden '
                        )

                    with ui.row():
                        feedback = None
                        for val in self.search_history:
                            if val['feedback'] and  val['feedback']['docno'] == record['docno']:
                                    feedback = val['feedback']['feedback']

                        toggle_value = None
                        if feedback:
                            toggle_value = 1 if feedback == '-' else 2 if feedback == '+' else None


                        ui.toggle(
                            options = {
                                1: 'Useless',
                                2: 'Relevant'
                            },
                            clearable = True,
                            value = toggle_value,
                            on_change = lambda e, record = record:
                                self.feedback(
                                    record = record,
                                    feedback_value = e.value
                                )
                        )

    def start(
            self
        ):
        """
        Start the GUI.
        """

        # BODYs
        self.search_title = ui.label(
                text = 'Clinical Trials Search Engine'
            ) \
            .classes(
                'text-h4 '
                'self-center '
                'text-center '
                'mt-8 '
                'mb-4 '
            )

        # SEARCH BAR
        with ui.row() \
            .classes(
                'flex '
                'justify-center '
                'items-center '
                'w-full '
                #'mb-4 '
            ):

            def _search_field(
                    text: str
                ):

                results = self.pipeline.transformer.autocomplete(text)
                self.search_field.set_autocomplete(results)

            self.search_field = ui.input(
                    placeholder = 'Search...',
                    on_change = lambda e:
                        _search_field(
                            text = e.value
                        )
                ) \
                .props(
                    'autofocus '
                    'outlined '
                    'rounded '
                ) \
                .classes(
                    'w-1/2 '
                    'self-center '
                )

            # SEARCH BUTTON
            self.search_button = ui.button(
                    icon = 'search',
                    on_click = lambda:
                        self.display(query = self.search_field.value)
                ) \
                .props(
                    'autofocus '
                    'rounded '
                ) \
                .classes(
                    'self-center '
                    'ml-2 '
                )

        # ADVANCED SEARCH
        async def change_q_exp():
            self.expansion = not self.expansion

        self.med_cond_label = ui.label('') \
            .classes(
                'text-h6 '
                'self-center '
                'text-center '
            )

        # RESULTS
        self.results = ui.column() \
            .classes(
                'mt-4 '
                'w-full '
            )

        # SETTINGS
        with ui.card() \
            .classes(
                'fixed '
                'bottom-4 '
                'right-4 '
            ):

            self.imp_q_switch = ui.switch(
                    text = "Expansion",
                    value = self.expansion,
                    on_change = lambda:
                        change_q_exp()
                ) \
                .props(
                    'color="green" '
                )

            self.mode_switch = ui.switch(
                    text = 'Dark Mode',
                    value = True
                ) \
                .props(
                    'color="red" '
                )

            self.dark_mode = ui.dark_mode() \
                .bind_value_from(self.mode_switch)

        try:
            ui.run(
                title = 'Clinical Trials SE',
                favicon = '🔎'
            )

        except Exception as e:
            self.error(f"Failed to run the Graphical Interface: {e}")