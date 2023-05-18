### Enhancing Cross-lingual Prompting with Dual Prompt Augmentation

This is a PyTorch implementation of our paper at ACL 2023 Findings: "Enhancing Cross-lingual Prompting with Dual Prompt Augmentation". 

- `run_{xnli,pawsx}.py`: python code for finetuning, modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py
- `{xnli,paws-x}.sh` shell scripts to run finetuning as a baseline
- `run-prompt.py`: python code for prompting with DPA
- `dpa-{xnli,pawsx}.sh`: shell scripts to run DPA training, different strategies can be played around by modifying parameters
- `balanced_data_processor.py`: for sampling a balanced training/dev set for few-shot experiments
- `prompt_helper.py`: some helper utils for prompting
- `modeling_xlmr.py`: the model class for prompting XLM-R
- `xnli-metrics.py`: an off-the-shelf script from `datasets`

- Requirements:
  - `transformers==4.10.3`
  - `datasets==1.12.1`
  - `torch==1.7.1`

