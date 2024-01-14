"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module includes the Extractor class and PickleExtractor class, designed to parse and extract information from XML files or a single large pickled file.
It supports processing of documents both sequentially and in parallel, extracting specific elements and text.
"""

# Standard Library
import os
import re
import html
import pickle
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-Party
from tqdm import tqdm
from typeguard import typechecked

# Local
from ..tools.logging import Logger

class XMLExtractor:

    @typechecked
    def __init__(
            self,
            verbose: bool = False
        ) -> None:
        """
        Initialize the parser with an optional folder filter.

        Args:
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
        self.data = []
        self.dataset = []

    @typechecked
    def _recurse_and_extract(
            self,
            element: ET.Element,
            text_pieces: List[str]
        ) -> str:
        """
        Recursively traverse the XML tree and extract text from elements.

        Args:
            element (ET.Element): The current XML element.
            text_pieces (List(str)): A list to accumulate the extracted text pieces.

        Returns:
            str: The concatenated text of all elements.
        """

        if element.text:
            cleaned_text = html.unescape(element.text.strip())
            cleaned_text = cleaned_text.replace('\r', ' ').replace('\n', ' ')
            cleaned_text = re.sub(' +', ' ', cleaned_text)  # Collapse multiple spaces into one
            text_pieces.append(cleaned_text)
        for child in element:
            self._recurse_and_extract(child, text_pieces)

        return ' '.join(text_pieces)

    @typechecked
    def _parse_xml(
            self,
            path: Union[str, os.PathLike]
        ) -> Dict[str, str]:
        """
        Parse the XML file at the given path and extract its text along with brieftitle and briefsummary.

        Args:
            path (Union[str, os.PathLike]): The path to the XML file.

        Returns:
            Dict[str, str]: The extracted text and specific elements from the XML file.
        """

        tree = ET.parse(path)
        root = tree.getroot()

        text_pieces = []
        brieftitle = ""
        briefsummary_pieces = []

        self._recurse_and_extract(root, text_pieces)

        brieftitle_element = root.find('.//brief_title')
        if brieftitle_element is not None:
            brieftitle = brieftitle_element.text.strip() if brieftitle_element.text else ""

        briefsummary_element = root.find('.//brief_summary')
        if briefsummary_element is not None:
            self._recurse_and_extract(briefsummary_element, briefsummary_pieces)
        briefsummary = ' '.join(briefsummary_pieces)

        return {
            'text': ' '.join(text_pieces),
            'title': brieftitle,
            'summary': briefsummary
        }

    @typechecked
    def process_file(
            self,
            file_info: Tuple[str, str]
        ) -> Dict[str, str]:
        """
        Process a single XML file and return its content.

        Args:
            file_info (Tuple[str, str]): A tuple containing the filename and folder name.

        Returns:
            Dict[str, str]: A dictionary with the document number and text.
        """

        filename, foldername = file_info
        file_path = os.path.join(foldername, filename)
        extracted_data = self._parse_xml(file_path)

        return {
            'docno': filename.removesuffix('.xml'),
            'text': extracted_data['text'],
            'title': extracted_data['title'],
            'summary': extracted_data['summary']
        }

    @typechecked
    def _iterate_folder(
            self,
            path: Union[str, os.PathLike]
        ) -> List[Tuple[str, str]]:
        """
        Iterate over folders and list XML files to be processed.

        Args:
            path (Union[str, os.PathLike]): The path to the directory containing folders of XML files.

        Returns:
            List[Tuple[str, str]]: A list of tuples with filenames and folder names.
        """

        all_files = []

        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)

            if os.path.isdir(folder_path) and folder.startswith('NCT'):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.xml') and filename.startswith('NCT'):
                        all_files.append((filename, folder_path))

        return all_files

    @typechecked
    def process_docs(
            self,
            folder_path: Union[str, os.PathLike],
            parallel_processing: bool = True
        ) -> List[Dict[str, str]]:
        """
        Process documents either in parallel or sequentially.

        Args:
            folder_path (Union[str, os.PathLike]): The path to the directory containing XML files.
            parallel_processing (bool): Flag to indicate parallel processing. Defaults to True.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with document numbers and texts.
        """

        all_files = self._iterate_folder(folder_path)

        results = []
        if parallel_processing:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_file, file_info) for file_info in all_files]

                with tqdm(total=len(all_files), desc="Processing files in parallel") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
        else:
            with tqdm(all_files, desc="Processing files sequentially") as pbar:
                for file_info in pbar:
                    result = self.process_file(file_info)
                    results.append(result)

        return results

class PickleExtractor:

    @typechecked
    def __init__(
            self,
            verbose: bool = False
        ) -> None:
        """
        Initialize the extractor for a large pickled data file.

        Args:
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
        self.data = []
        self.dataset = []

    @typechecked
    def _load(
            self,
            file_path: Union[str, os.PathLike]
        ) -> None:
        """
        Load and return data from a pickled file.

        Args:
            file_path (Union[str, os.PathLike]): The path to the pickled file.
        """

        with open(file_path, 'rb') as file:
            while True:
                try:
                    document = pickle.load(file)
                    self.data.extend(document)
                except EOFError:
                    break

    @typechecked
    def _convert(
            self,
            element: Dict[str, str]
        ) -> ET.Element:
        """
        Convert a dictionary to an XML element.

        Args:
            element (Dict[str, str]): The dictionary to convert.

        Returns:
            ET.Element: The converted XML element.
        """

        root = ET.Element("clinical_study")

        for key, value in element.items():
            child = ET.SubElement(root, key)
            child.text = str(value)

        return root

    @typechecked
    def _write(
            self,
            document: Dict[str, str],
            save_path: Union[str, os.PathLike]
        ) -> None:
        """
        Write a document as an XML file.

        Args:
            document (Dict[str, str]): The document to be written.
            save_path (Union[str, os.PathLike]): The path to save the XML file.
        """

        try:
            element = self._convert(document)
            tree = ET.ElementTree(element)
            tree.write(
                os.path.join(save_path, document['nct_id'] + '.xml'),
                encoding = 'utf-8',
                xml_declaration = True
            )

        except Exception as e:
            self.warning(f"Failed saving of document {document['nct_id']}: {e}")

    @typechecked
    def process_data(
            self,
            file_path: Union[str, os.PathLike],
            save_path: Union[str, os.PathLike],
            save: bool = True,
            parallel_processing: bool = True,
            n_max: int = -1
        ) -> List[Dict[str, str]]:
        """
        Process and save data from the pickled file.

        Args:
            file_path (Union[str, os.PathLike]): Path to the pickled file.
            save_path (Union[str, os.PathLike]): Path to save the processed XML files.
            save (bool): If True, save the documents. Defaults to True.
            parallel_processing (bool): If True, processes the data in parallel. Defaults to True.
            n_max (int): The maximum number of documents to process. If -1, process all documents. Defaults to -1.

        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the processed data.
        """

        self._load(file_path)

        if save:
            if parallel_processing:
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for document in tqdm(self.data[:n_max], desc="Processing documents"):
                        futures.append(executor.submit(self._write, document, save_path))

                    for future in tqdm(as_completed(futures), total=len(futures), desc="Saving documents"):
                        future.result()

            else:
                for document in self.data:
                    self._write(document, save_path)

        self.dataset = []

        for document in self.data:
            try:
                text = ' '.join(str(value) for key, value in document.items() if isinstance(value, str))

                processed_doc = {
                    'docno': document.get('nct_id'),
                    'text': text,
                    'title': document.get('brief_title', ''),
                    'summary': document.get('brief_summary', '')
                }
                self.dataset.append(processed_doc)

            except Exception as e:
                self.warning(f"Failed conversion for document {document.get('nct_id')}: {e}")

        return self.dataset


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")