"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This script initializes and starts the graphical user interface (GUI) of the CTSE project.
It sets the directory path for the GUI and begins the GUI operation.
"""

from _setup import check
check(verbose=True)

from src.gui import GUI

# from src.tools.utils import import_paths
# paths = import_paths('path\\to\\ClinicalTrialsSE\\src\\paths.txt')

paths = {
    'DATA' : "path\\to\\ClinicalTrialsSE\\data",
    'DATASET' : "path\\to\\ClinicalTrialsSE\\data\\TREC21_processed.pkl",
    'DOCUMENTS' : "path\\to\\ClinicalTrialsSE\\data\\TREC21",
    'INDEXING_FILES' : "path\\to\\ClinicalTrialsSE\\data\\index",
    #'LLM' : "path\\to\\Llama2\\llama-2-7b-chat.Q5_K_M.gguf",
    'LLM' : "path\\to\\Mistral\\mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    'BERT' : "path\\to\\ClinicalBERT",
    'JDK' : "path\\to\\Java\\jdk-21\\bin",
    'EVAL' : "path\\to\\ClinicalTrialsSE\\data\\eval",
    'LOG' : "path\\to\\ClinicalTrialsSE\\data\\log.txt"
}

gui = GUI(
    paths = paths,
    verbose = True
)

gui.start()