# DPA: Dual Prompt Augmentation for Cross-lingual Transfer

- `run_{xnli,pawsx}.py` for finetuning
- `{xnli,paws-x}.sh` shell scripts for finetuning
- `run-prompt.py` prompting with DPA for XNLI/PAWS-X
- `dpa-{xnli,pawsx}.sh` run DPA training, different strategies can be played around by modifying parameters
- `balanced_data_processor.py` for sampling a balanced training/dev set
- `prompt_helper.py` some helper functionality for prompting
- `modeling_xlmr.py` the model class for prompting XLM-R
- `xnli-metrics.py` an off-the-shelf script from `datasets`

- Requirements:
  - `transformers==4.10.3`
  - `datasets==1.12.1`
  - `torch==1.7.1`

