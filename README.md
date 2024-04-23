# Overview
The code helps record all the communication with LLMs by a human. This repo is a simple implementation but can be adapted according to the user's needs. Currently, this code will log the date, time, input prompt, output from LLM, model names and set of hyperparameters when the user communicates with LLM using some prompt in a json file. To check the log files, go to Logs/ which holds datewise folders and respective json files (contains the before mentioned info)
# Usage
Update your hugging face access token in `.env` file and run the `main.py` file to log inputs/outputs

Tune the hyperparameters or change the model id's in `config.ini` file, if needed

To use the different LLM, update the `load_llm` function according to the needs in `helper_functions.py`
# License
The repository is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
# Citation
```
@misc{LLM_Logs,
  author = {Vamsi Kommineni},
  month = {04},
  title = {{LLM_Logs}},
  url = {https://github.com/Vamsi-Kommineni/LLM_Logs},
  year = {2024}
}
```
