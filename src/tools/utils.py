"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes utility functions for package management and data processing.
It provides functionality to process search result data frames and open xml files for visualization.
"""

# Standard Library
import os
import re
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from typing import Union, List, Tuple

# Third-Party
import pandas as pd
from typeguard import typechecked

# Local
from .logging import Logger


logger = Logger(
    id = 'utils',
    log_file_path = None,
    raise_errors = True,
    verbose = True
)

@typechecked
def import_paths(
        file_path: Union[str, os.PathLike]
    ) -> None:
    """
    Import paths from a given file.

    Args:
        file_path (Union[str, os.PathLike]): Path to the file containing paths.

    Returns:
        Dict[str, str]: A dictionary of paths.
    """

    try:
        paths = {}

        path_pattern = re.compile(r'^\s*([A-Z_]+)\s*=\s*"([^"]+)"\s*$')

        with open(file_path, 'r') as file:
            for line in file:
                match = path_pattern.match(line)
                if match:
                    key = match.group(1)
                    path = match.group(2)
                    paths[key] = path

    except Exception as e:
        logger.warning(f"Failed to import paths: {e}")

    return paths

@typechecked
def get_topn(
        results: pd.DataFrame,
        n: int,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
    """
    Retrieves the top N documents from a DataFrame of search results based on their scores.

    Args:
        results (pd.DataFrame): A DataFrame containing search results.
        n (int): The number of top documents to retrieve.
        verbose (bool): If True, additional details are printed during operations. Defaults to False.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing top N document IDs and their scores.
    """

    n = min(n, len(results))

    top_n_results = results.sort_values(
        by = 'score',
        ascending = False
    ).head(n)

    top_n_tuples = [(row['docno'], row['score']) for index, row in top_n_results.iterrows()]

    if verbose:
        _results: str = ''
        for doc_id, score in top_n_tuples:
            _results += f"\nID: {doc_id} - Score: {score}"
        logger.info(f"Topn documents:{_results}\n")

    return top_n_tuples

@typechecked
def _convert_to_html(
        root: ET.Element
    ) -> str:
    """
    Convert the XML content to HTML format by iterating through all elements.

    Args:
        root (ET.Element): The root element of the XML tree.

    Returns:
        str: The converted HTML content as a string.
    """

    html_parts = ['<html><head><title>Clinical Study</title><style> .tag { color: blue; } </style></head><body>']

    def _process_xml(
            element,
            depth: int = 0
        ) -> None:

        tag = 'h' + str(depth + 1) if depth < 3 else 'p'

        if depth > 0:
            tag_name = ' '.join(word.capitalize() for word in element.tag.replace('_', ' ').split())
            html_parts.append(f"<{tag}><span class='tag'>{tag_name}:</span> ")

        if element.text and element.text.strip():
            html_parts.append(element.text.strip())

        for child in element:
            _process_xml(child, depth + 1)

        if depth > 0:
            html_parts.append(f"</{tag}>")

    _process_xml(root)
    html_parts.append('</body></html>')

    return ''.join(html_parts)

@typechecked
def open_xml(
        path: Union[str, os.PathLike]
    ) -> None:
    """
    Given an XML file, convert it to HTML and open it in the browser.

    Args:
        path (Union[str, os.PathLike]): The path to the XML file.
    """

    try:
        tree = ET.parse(path)
        root = tree.getroot()
        html_content = _convert_to_html(root)

        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as temp_file:
            temp_file.write(html_content)
            temp_file.flush()

        temp_file_path = os.path.abspath(temp_file.name).replace('\\', '/')
        subprocess.run(f'start {temp_file_path}', shell=True)

    except Exception as e:
        logger.warning(f"Failed to open \"{path}\": {e}")


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")