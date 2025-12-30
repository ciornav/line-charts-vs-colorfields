# Reproducing the results of the pilots and of the study comparing line charts to colorfields
![](/data/colorfields_vs_line_charts.png)

## Goal of this repository
Allow other researchers to reproduce (and reuse) the study carried out for comparing line charts and colorfields as **scents** in a multi-input, multi-output context.

## Set up
1. Install python 3.11.1
   - https://www.python.org/downloads/release/python-3111/
2. Install poetry (project was tested with version 1.4.2). This is a package manager which will install all required python libraries for the project to run.
   - https://python-poetry.org/docs/
   - in case newer version don't work the exact version of poetry can be located here:  https://github.com/python-poetry/poetry/releases/tag/1.4.2
3. Clone this project
4. In a terminal, change directory for this project's root
5. Install a virtual environment in the project's root
   - Windows: python -m venv .venv
   - Linux: python3 -m venv .venv
   * Naming the virtual environment .venv will help poetry to locate the virtual environment and install the required packages directly in this venv
6. Run `poetry install` to install all required packages into your newly created venv
## Running the project

7. Run with python any of the following 3 scripts to reproduce the results:
   * `colormaps_pilot_analyzer.py` # for the colormap pilot
   * `height_pilot_analyzer.py` # for the height pilot
   * `study_analyzer.py` # for the study itself (a little slower compared to the pilots)
8. Each script contains a **main** function (towards the end of the file). In this function each task (out of 4) has its own analysis and plotting. It might be useful to breakpoint after each task analysis to investigate the returned results and respective plots. Otherwise the results are simply printed in the console.

## Project contents
Below directory structure is used for this project. The structure allows to locate the 3 scripts previously mentioned and have some information about each subdirectory and it's role.
The comments next to each file are useful for understanding which files are used for which task.

```plaintext
project_root/
├── data                                                  # Data files for the pilots and the study itself              
│   ├── colormap_pilot_data
│   │   └── data_files_for_colormap_pilot...
│   ├── height_pilot_data
│   │   ├── data_files_for_height_pilot...
│   └── study_data
│       ├── nadir
│       │   ├── files_useful_for_starting_best_points...
│       ├── participant_data.csv
├── helpers                                                # Helper scripts for the pilots and the study                
│   ├── __init__.py
│   ├── colormap_pilot_helpers.py
│   ├── common_helpers.py
│   ├── height_pilot_helpers.py
│   └── study_helpers
│       ├── __init__.py
│       ├── general_study_helpers.py
│       └── moo_helpers.py
├── .gitignore                                             # Git ignore file
├── colormaps_pilot_analyzer.py                            # ====This script allows to reproduce the plots and results from the colormap pilot.====
├── height_pilot_analyzer.py                               # ====This script allows to reproduce the plots and results from the height pilot.====
├── poetry.lock                                            # File used by poetry to install the required packages (and dependencies) at their exact versions
├── pyproject.toml                                         # File used by poetry to configure the project
├── README.MD                                              # Project documentation
└── study_analyzer.py                                      # ====This script allows to reproduce the plots and results from the main study.====
 