"""
University of Pavia
Artificial Intelligence BSc
Information Retrieval and Recommender Systems

Authors:
- Michele Ventimiglia
- Manuel Dellabona

This script is part of the Clinic Trials SE project and is released under the GNU General Public License:
https://www.gnu.org/licenses/gpl-3.0.html#license-text

This module checks for virtual environment and set project path for modules import.
"""

# Standard Library
import sys
from pathlib import Path


def check(
        verbose: bool = False
    ) -> None:
    """
    Check for virtual environment and set project path.

    Args:
        verbose (bool): If True, additional details are printed during operations. Defaults to False.
    """

    if verbose:
        print("\033[94m(i)\033[0m [SETUP] Checking for virtual environment...")

    if sys.prefix == sys.base_prefix:
        print("\033[31mERROR\033[0m: [SETUP] Virtual environment not active!")
        raise

    if verbose:
        print("\033[32mSUCCESS\033[0m: [SETUP] Virtual environment detected!")

    try:
        path_root = Path(__file__).parents[1]

        if not path_root in sys.path:
            sys.path.append(str(path_root))

            if verbose:
                print("\033[94m(i)\033[0m [SETUP] Path root added to the environment variables.")

    except Exception as e:
        print(f"\033[31mERROR\033[0m: [SETUP] Failed to set main folder: {e}")
        raise

    if verbose:
        print("\033[32mSUCCESS\033[0m: [SETUP] Check completed!")


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")