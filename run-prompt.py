#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

# from transformers.utils.dummy_pt_objects import AutoModelForMaskedLM
from transformers.utils.versions import require_version

from balanced_data_processor import balanced_sampler
from prompt_helper import Prompt_Helper
from modeling_xlmr import PromptXLMR

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.11.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset: str = field(
        default="paws-x", metadata={"help": "Which dataset to use. {paws-x/xnli}"}
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=8,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=8,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    server_ip: Optional[str] = field(
        default=None, metadata={"help": "For distant debugging."}
    )
    server_port: Optional[str] = field(
        default=None, metadata={"help": "For distant debugging."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="./xlm-roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    pretrain_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to the model trained on the English dataset, specify this when evaluating on other languages"
        },
    )
    language: str = field(
        default="en,de",
        metadata={
            "help": "Evaluation language. Also train language if `train_language` is set to None."
        },
    )
    train_language: Optional[str] = field(
        default="en",
        metadata={
            "help": "Train language if it is different from the evaluation language."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="./cache-data",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={
            "help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_soft_prompt: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use soft prompting without tuning the LM."
        },
    )
    prompt_length: int = field(
        default=60,
        metadata={
            "help": "The prompt length for soft prompting without tuning the parameter of the model."
        },
    )
    tune_LM: bool = field(
        default=True,
        metadata={"help": "Whether to tune the LM or just perform prompt tuning."},
    )
    separate_lan_label_word: bool = field(
        default=False,
        metadata={
            "help": "Whether use translated label word when testing in another language"
        },
    )
    multi_lingual_optim: bool = field(
        default=False,
        metadata={"help": "Whether or not to use multilingual label word to optimize"},
    )
    multi_lingual_label_word: bool = field(
        default=False,
        metadata={
            "help": "Whether make predictions by taking the sum of label words from 15 languages"
        },
    )
    mixup_strategy: str = field(
        default="hidden",
        metadata={
            "help": "hidden/input_embedding"
        },
    )
    mixup_alpha: float = field(
        default=0.5,
        metadata={
            "help": "mix up alpha parameter, the smaller the alpha is, the more concentrated in double sides between [0, 1]"
        },
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(data_args.server_ip, data_args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if training_args.do_train:
        if model_args.train_language is None:
            raise ValueError("Train language args should not be None for clarity!")
        else:
            train_dataset = load_dataset(
                f"{data_args.dataset}",
                model_args.train_language,
                split="train",
                cache_dir=model_args.cache_dir,
            )

        label_list = predict_dataset.features["label"].names

    if training_args.do_eval:
        # the language of development set should be the same as the train_language
        eval_dataset = load_dataset(
            f"{data_args.dataset}",
            model_args.train_language,
            split="validation",
            cache_dir=model_args.cache_dir,
        )
        
        label_list = predict_dataset.features["label"].names

    if training_args.do_predict:
        test_languages = model_args.language.split(",")
        predict_datasets = []
        for test_language in test_languages:
            predict_dataset = load_dataset(
                f"{data_args.dataset}",
                test_language,
                split="test",
                cache_dir=model_args.cache_dir,
            )
            label_list = predict_dataset.features["label"].names
            predict_datasets.append(predict_dataset)


    # Labels
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    prompt_helper = Prompt_Helper(data_args.dataset, tokenizer)

    # model = AutoModelForMaskedLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    if not model_args.use_soft_prompt and not model_args.tune_LM:
        raise ValueError(
            "If you don't tune LM and use a discrete prompt, then there's no parameter to update"
        )
    model = PromptXLMR(
        model_args,
        config,
        prompt_helper,
        use_soft_prompt=model_args.use_soft_prompt,
        prompt_length=model_args.prompt_length,
        tune_LM=model_args.tune_LM,
        multi_lingual_optim=model_args.multi_lingual_optim,
        multi_lingual_label_word=model_args.multi_lingual_label_word,
        mixup_strategy=model_args.mixup_strategy,
        mixup_alpha=model_args.mixup_alpha,
    )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        prompts = prompt_helper.convert_examples_to_prmopts(examples)
        # convert the classification labels to MaskedLM labels
        examples["label"] = prompt_helper.get_prompt_labels(prompts, examples["label"])
        # rename the key of label for the forward pass of automlmmodel
        examples["labels"] = examples.pop("label")
        # Tokenize the texts
        return tokenizer(
            prompts,
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            indices = balanced_sampler(
                train_dataset, data_args.max_train_samples, label_list
            )
            train_dataset = train_dataset.select(indices)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            prompt_helper.update_verbalizer(
                model_args.train_language, model_args.separate_lan_label_word
            )
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                keep_in_memory=True,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), num_labels):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # Load from the pretrained prompting model
        if model_args.pretrain_model_path is not None:
            logger.info(
                f"Initialize the model using the pretrained model saved at {model_args.pretrain_model_path}"
            )
            model.load_state(model_args.pretrain_model_path)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            indices = balanced_sampler(
                eval_dataset, data_args.max_eval_samples, label_list
            )
            eval_dataset = eval_dataset.select(indices)
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            prompt_helper.update_verbalizer(
                model_args.train_language, model_args.separate_lan_label_word
            )
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                keep_in_memory=True,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            for i, predict_dataset in enumerate(predict_datasets):
                predict_datasets[i] = predict_datasets[i].select(
                    range(data_args.max_predict_samples)
                )
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            for i, (predict_dataset, test_language) in enumerate(
                zip(predict_datasets, test_languages)
            ):
                # update the verbalizer to preprocess the data of the test language
                prompt_helper.update_verbalizer(
                    test_language, model_args.separate_lan_label_word
                )
                predict_datasets[i] = predict_datasets[i].map(
                    preprocess_function,
                    batched=True,
                    keep_in_memory=True,
                    desc="Running tokenizer on prediction dataset",
                )
            # switch back to English to perform training

    # Get the metric function
    metric = load_metric("xnli_metrics.py")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        # convert the masked_lm labels back to normal classification labels
        labels = prompt_helper.get_cls_label_from_prompt_labels(p.label_ids)
        return metric.compute(predictions=preds, references=labels)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = DataCollatorForTokenClassification(
            tokenizer, max_length=data_args.max_seq_length
        )
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        prompt_helper.update_verbalizer(
            model_args.train_language, model_args.separate_lan_label_word
        )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        prompt_helper.update_verbalizer(
            model_args.train_language, model_args.separate_lan_label_word
        )
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        for predict_dataset, test_language in zip(predict_datasets, test_languages):
            prompt_helper.update_verbalizer(
                test_language, model_args.separate_lan_label_word
            )
            predictions, labels, metrics = trainer.predict(
                predict_dataset, metric_key_prefix="{}_predict".format(test_language)
            )

            max_predict_samples = (
                data_args.max_predict_samples
                if data_args.max_predict_samples is not None
                else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            trainer.log_metrics("{}_predict".format(test_language), metrics)
            trainer.save_metrics("{}_predict".format(test_language), metrics)

            predictions = np.argmax(predictions, axis=1)
            output_predict_file = os.path.join(
                training_args.output_dir, "{}_predictions.txt".format(test_language)
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
