# ClinicalTrialsSE

Clinical Trials Search Engine for Information Retrieval and Recommender System Project

---

# Disclaimer

This project has been developed for academical purposes only.

This project is released under the [GNU General Public Licence](https://github.com/MikiTwenty/ClinicalTrialsSE/LICENCE).

---

# Setup

Clone the repository locally:

- **Git** : ```git clone https://github.com/MikiTwenty/ClinicalTrialsSE```
- **GitHub** : ```gh repo clone MikiTwenty/ClinicalTrialsSE```

## Requirements

- **Virtual Environment** with Python >= 3.10
- Required **Python packeges** (see [requirements.txt](https://github.com/MikiTwenty/ClinicalTrialsSE/rquirements.txt))
- **Java SDK**

### Download Clinical Bert Model

Get files from Hugging Face:

- ```git lfs install```
- ```git clone https://huggingface.co/medicalai/ClinicalBERT```

### Download LLM GGUF Model files

You can find the models on [Hugging Face](https://huggingface.co/TheBloke).
Suggested models are **models with 7B parameters** such as:

- [Llama 2 Chat 7B GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [Mistral Instruct 7B v0.2 GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

### Enable GPU computing (Optional)

- **PyTorch** : see [documentation](https://pytorch.org/get-started/locally/)
- **llama-cpp-python** : see [documentation](https://github.com/abetlen/llama-cpp-python/)

---

# Usage

Initialize the Graphical User Interface to run the SE.

## src.gui.```GUI```

Initialize the **Graphical User Interface** (powered by [NiceGUI](https://nicegui.io/)).

**Parameters**

- **```paths```** ```(Optional[Dict[str, str]])```: A dictionary of paths if already available.

    The paths keys must be:

    ```'DATA'```: the folder containing all the data;

    ```'DATASET'```: the dataset.pkl file;

    ```'DOCUMENTS'```: the folder containing all the dataset files;

    ```'INDEXING_FILES'```: the folder for the indexing files;

    ```'LLM'```: the folder containg the LLM GGUF files;

    ```'BERT'```: the folder containg ClinicalBert model files;

    ```'JDK'```: the folder containing Java SDK;

    ```'EVAL'```: the folder containing the evaluation files (optional);

    ```'LOG'```: the log.txt file (optional).

- ```verbose``` ```(bool)```: If ```True```, enables verbose output.

### GUI.```start()```

Initialize the SE and start the GUI.

---

# Info

- Univeristy of Pavia
- Artificial Intelligence BSc
- Information Retrieval and Recommender Systems
- Authors: Michele Ventimiglia (@MikiTwenty), Manuel Dellabona (@manudella)
