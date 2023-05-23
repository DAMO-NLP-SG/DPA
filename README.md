# Enhancing Cross-lingual Prompting with Dual Prompt Augmentation

This is a PyTorch implementation of our paper at ACL 2023 Findings: "Enhancing Cross-lingual Prompting with Dual Prompt Augmentation". 

### Shell Scripts

- `dpa-{xnli,pawsx}.sh`: shell scripts to run DPA training, different strategies can be played around by modifying parameters
- `{xnli,paws-x}.sh` shell scripts to run finetuning as a baseline

### Python Codes

- `run_{xnli,pawsx}.py`: python code for finetuning, modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_xnli.py
- `run-prompt.py`: python code for prompting with DPA
- `balanced_data_processor.py`: for sampling a balanced training/dev set for few-shot experiments
- `prompt_helper.py`: some helper utils for prompting
- `modeling_xlmr.py`: the model class for prompting XLM-R
- `xnli-metrics.py`: an off-the-shelf script from `datasets`

### Dependencies

- `transformers==4.10.3`
- `datasets==1.12.1`
- `torch==1.7.1`

### Citation

If you find our code useful, please cite the following, thank you!

````
```bibtxt
@inproceedings{acl23/DPA,
  author    = {Meng Zhou and
               Xin Li and
               Yue Jiang and
               Lidong Bing},
  title     = {Enhancing Cross-lingual Prompting with Dual Prompt Augmentation},
  booktitle = {Findings of the 2023 ACL},
  year      = {2023},
  url       = {},
}
```
````
