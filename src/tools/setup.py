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
It provides functionality to check for pip package dependencies and install packages.
"""

# Standard Library
import os
import sys
import platform
from typing import Union

# Third-Party
from typeguard import typechecked

# Local
from .logging import Logger


logger = Logger(
    id = 'Setup',
    log_file_path = None,
    raise_errors = True,
    verbose = True
)

def _get_python_command() -> str:
    """
    Determines the correct Python command based on the operating system.

    Returns:
        str: A string of the Python command.
    """

    return "python3" if platform.system() != "Windows" else "python"

def _check_requirements() -> None:
    """
    Checks for installed pip package dependencies using the pip check command.
    """

    python_command = _get_python_command()
    command = f"{python_command} -m pip check"
    os.system(command)

def _purge_cache() -> None:
    """
    Purges the pip cache to free up space and remove outdated packages.
    """

    python_command = _get_python_command()
    command = f"{python_command} -m pip cache purge"
    os.system(command)

@typechecked
def _install_package(
        package_name: str,
        upgrade: bool = False,
        quiet: bool = False
    ) -> None:
    """
    Installs a package using pip.

    Args:
        package_name (str): The name of the package to install.
        upgrade (bool): Whether to upgrade the package if it's already installed. Defaults to False.
        quiet (bool): Suppresses output from the pip command. Defaults to False.
    """

    python_command = _get_python_command()
    upgrade_str = "--upgrade" if upgrade else ""
    quiet_str = "--quiet" if quiet else ""
    command = f"{python_command} -m pip install {upgrade_str} {quiet_str} {package_name}"
    os.system(command)

@typechecked
def check_requirements(
        req_file_path: Union[str, os.PathLike],
        upgrade: bool = False,
        quiet: bool = False
    ) -> None:
    """
    Installs or upgrades packages listed in the requirements file and purges pip cache.

    Args:
        req_file_path (Union[str, os.PathLike]): Path to the requirements file.
        upgrade (bool): If True, upgrade packages listed in the requirements file. Defaults to False.
        quiet (bool): If True, suppress output from pip commands. Defaults to False.
    """

    if not os.path.exists(req_file_path):
        logger.warning(f"Requirements file does not exist: {req_file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(req_file_path, 'r') as req_file:
            for line in req_file:
                package_name = line.strip()
                if package_name:
                    _install_package(package_name, upgrade, quiet)

        _check_requirements()
        _purge_cache()

    except Exception as e:
        logger.warning(f"Failed to check requirements: {e}")

    logger.success("Requirements check done")


if __name__ == '__main__':
    print("\033[33mWARNING\033[0m: This script should not be run as main!")