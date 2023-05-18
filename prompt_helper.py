import torch
import numpy as np
from transformers import PreTrainedTokenizer
from special_tokens import *


class Prompt_Helper:
    def __init__(self, dataset: str, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.global_verbalizers = self._setup_global_verbalizers(dataset)
        self.text_a_key, self.text_b_key = self._setup_dataset_key(dataset)
        self.inverted_global_verbalizers = {
            lan: {v: k for k, v in self.global_verbalizers[lan].items()}
            for lan in self.global_verbalizers.keys()
        }
        # The default verbalizer is set to English for training
        self.verbalizers = self.global_verbalizers["en"]
        self.inverted_verbalizers = {v: k for k, v in self.verbalizers.items()}

        self.num_classes = len(self.global_verbalizers["en"].keys())

        self.tokenizer = tokenizer

    def _setup_dataset_key(self, dataset: str):
        if dataset == "xnli":
            return "premise", "hypothesis"
        elif dataset == "paws-x":
            return "sentence1", "sentence2"
        else:
            raise ValueError(f"The {dataset} dataset is not supported for now!")

    """
    Set up the verbalizer according to the dataset you run
    Note that the label should be listed from 0 to 1, 2
    to keep consistent with convert_mlm_prediction_scores_to_seqcls
    """

    def _setup_global_verbalizers(self, dataset: str):
        if dataset == "xnli":
            # the label word for neutral is translated by "maybe" or "possibly"
            # global_verbalizers = {
            #     "ar": {AR_YES: 0, AR_MAYBE: 1, AR_NO: 2},
            #     "bg": {"да": 0, "може": 1, "не": 2},
            #     "de": {"Ja": 0, "möglicherweise": 1, "Nein": 2},
            #     "el": {"Μάλιστα": 0, "μπορεί": 1, "όχι": 2},
            #     "en": {"yes": 0, "maybe": 1, "no": 2},
            #     "es": {"sí": 0, "talvez": 1, "no": 2},
            #     "fr": {"Oui": 0, "possible": 1, "non": 2},
            #     "hi": {"हां": 0, "शायद": 1, "न": 2},
            #     "ru": {"да": 0, "може": 1, "нет": 2},
            #     "sw": {"ndio": 0, "labda": 1, "sio": 2},
            #     "th": {"ใช่": 0, "อาจจะ": 1, "ไม่": 2},
            #     "tr": {"Evet": 0, "belki": 1, "hiçbir": 2},
            #     "ur": {UR_YES: 0, UR_MAYBE: 1, UR_NO: 2},
            #     "vi": {"dạ": 0, "lẽ": 1, "không": 2},
            #     "zh": {"是": 0, "也许": 1, "否": 2},
            # }
            global_verbalizers = {
                "ar": {AR_YES: 0, AR_MAYBE: 1, AR_NO: 2},
                "bg": {"да": 0, "може": 1, "не": 2},
                "de": {"Ja": 0, "möglicherweise": 1, "Nein": 2},
                "el": {"Μάλιστα": 0, "μπορεί": 1, "όχι": 2},
                "en": {"yes": 0, "maybe": 1, "no": 2},
                "es": {"sí": 0, "talvez": 1, "no": 2},
                "fr": {"Oui": 0, "possible": 1, "non": 2},
                "hi": {"हां": 0, "शायद": 1, "न": 2},
                "ru": {"да": 0, "може": 1, "нет": 2},
                "sw": {"ndio": 0, "labda": 1, "sio": 2},
                "th": {"ใช่": 0, "อาจจะ": 1, "ไม่": 2},
                "tr": {"Evet": 0, "belki": 1, "hiçbir": 2},
                "ur": {UR_YES: 0, UR_MAYBE: 1, UR_NO: 2},
                "vi": {"dạ": 0, "lẽ": 1, "không": 2},
                "zh": {"是": 0, "也许": 1, "否": 2},
            }
        elif dataset == "paws-x":
            global_verbalizers = {
                "de": {"Nein": 0, "Ja": 1},
                "en": {"no": 0, "yes": 1},
                "es": {"no": 0, "sí": 1},
                "fr": {"non": 0, "Oui": 1},
                "zh": {"否": 0, "是": 1},
                "ja": {"ない": 0, "はい": 1},
                "ko": {"아니": 0, "예": 1},
            }
        else:
            raise ValueError(f"The {dataset} dataset is not supported for now!")
        return global_verbalizers

    """
    Update the verbalizer to a certain language
    This is used when performing zero-shot transfer to another language and preprocessing the test data
    """

    def update_verbalizer(self, lan: str, multi_label_word_testing: bool) -> None:
        if multi_label_word_testing:
            if lan not in self.global_verbalizers.keys():
                raise ValueError(f"The verbalizer for this language ({lan}) is not defined")
            self.verbalizers = self.global_verbalizers[lan]
        else:
            self.verbalizers = self.global_verbalizers["en"]
        self.inverted_verbalizers = {v: k for k, v in self.verbalizers.items()}

    """
    Convert the prediction output of the LM to the classification label
    e.g. [-100 ... TOKENIZER_ID(yes) -100 ...] -> [0] (entailment)
    """

    def get_cls_label_from_prompt_labels(self, prompt_labels: torch.Tensor):
        cls_labels = prompt_labels[prompt_labels != -100]
        for i, cls_label in enumerate(cls_labels):
            verb = self.tokenizer.decode(int(cls_label))
            if verb in self.verbalizers.keys():
                cls_labels[i] = self.verbalizers[verb]
            else:
                raise ValueError("Class label is not in the verbalizer?")
        return cls_labels

    """
    Convert the output of the mlm head to an output like a classification head
    The prediction score of a classification head is taken from the output of mlm head directly
    e.g. prediction_scores: [8, 256, 250002] -> converted_prediction_scores: [8, 3]
    This function requires the verbalizer defined in the correct order (0, 1, ...)
    """

    def convert_mlm_prediction_scores_to_seqcls(
        self, prediction_scores: torch.Tensor, labels: torch.Tensor,
    ):
        converted_prediction_scores = torch.zeros(
            prediction_scores.shape[0], self.num_classes
        ).to(prediction_scores.device)
        target_token_indices = torch.where(labels != -100)[1]
        verbal_indices = [
            self.tokenizer.encode(verbal)[-2] for verbal in self.verbalizers.keys()
        ]
        for i, target_token_index in enumerate(target_token_indices):
            for j, verbal_index in enumerate(verbal_indices):
                converted_prediction_scores[i, j] = prediction_scores[
                    i, target_token_index, verbal_index
                ]
        return converted_prediction_scores

    """
    Convert the output of the mlm head to an output like a classification head
    The prediction score of a classification head is taken from the output of mlm head directly
    This function uses the sum of label words in 15 languages compared with convert_mlm_prediction_scores_to_seqcls
    e.g. prediction_scores: [8, 256, 250002] -> converted_prediction_scores: [8, 3]
    """

    def convert_mlm_prediction_scores_to_seqcls_multilingual(
        self, prediction_scores: torch.Tensor, labels: torch.Tensor,
    ):
        converted_prediction_scores = torch.zeros(
            prediction_scores.shape[0], self.num_classes
        ).to(prediction_scores.device)
        target_token_indices = torch.where(labels != -100)[1]
        for lan in self.global_verbalizers.keys():
            verbal_indices = [
                self.tokenizer.encode(verbal)[-2]
                for verbal in self.global_verbalizers[lan].keys()
            ]
            for i, target_token_index in enumerate(target_token_indices):
                for j, verbal_index in enumerate(verbal_indices):
                    converted_prediction_scores[i, j] += torch.exp(
                        prediction_scores[i, target_token_index, verbal_index]
                    )
        return converted_prediction_scores

    """
    Convert the output of the mlm head to an output like a classification head
    The prediction score of a classification head is taken from the output of mlm head directly
    This function uses the largest value in 15 languages compared with convert_mlm_prediction_scores_to_seqcls
    e.g. prediction_scores: [8, 256, 250002] -> converted_prediction_scores: [8, 3]
    """

    def convert_mlm_prediction_scores_to_seqcls_maximum(
        self, prediction_scores: torch.Tensor, labels: torch.Tensor,
    ):
        converted_prediction_scores = torch.zeros(
            prediction_scores.shape[0], self.num_classes
        ).to(prediction_scores.device)
        target_token_indices = torch.where(labels != -100)[1]
        for lan in self.global_verbalizers.keys():
            verbal_indices = [
                self.tokenizer.encode(verbal)[-2]
                for verbal in self.global_verbalizers[lan].keys()
            ]
            for i, target_token_index in enumerate(target_token_indices):
                for j, verbal_index in enumerate(verbal_indices):
                    if (
                        prediction_scores[i, target_token_index, verbal_index]
                        > converted_prediction_scores[i, j]
                    ):
                        converted_prediction_scores[i, j] = prediction_scores[
                            i, target_token_index, verbal_index
                        ]
        return converted_prediction_scores

    """
    Assumption: separate_lan_label_word should be set to True
    Convert the output of the mlm head to an output like a classification head
    The prediction score of a classification head is taken from the output of mlm head directly
    This function uses the sum of label words in 2 languages compared with convert_mlm_prediction_scores_to_seqcls
    e.g. prediction_scores: [8, 256, 250002] -> converted_prediction_scores: [8, 3]
    """

    def convert_mlm_prediction_scores_to_seqcls_bilingual(
        self, prediction_scores: torch.Tensor, labels: torch.Tensor,
    ):
        converted_prediction_scores = torch.zeros(
            prediction_scores.shape[0], self.num_classes
        ).to(prediction_scores.device)
        target_token_indices = torch.where(labels != -100)[1]
        # English label word
        verbal_indices = [
            self.tokenizer.encode(verbal)[-2]
            for verbal in self.global_verbalizers["en"].keys()
        ]
        for i, target_token_index in enumerate(target_token_indices):
            for j, verbal_index in enumerate(verbal_indices):
                converted_prediction_scores[i, j] += torch.exp(
                    prediction_scores[i, target_token_index, verbal_index]
                )
        # Language-specific label word
        verbal_indices = [
            self.tokenizer.encode(verbal)[-2] for verbal in self.verbalizers.keys()
        ]
        for i, target_token_index in enumerate(target_token_indices):
            for j, verbal_index in enumerate(verbal_indices):
                converted_prediction_scores[i, j] += torch.exp(
                    prediction_scores[i, target_token_index, verbal_index]
                )
        return converted_prediction_scores

    """
    Derive the mlm label for all the lanaguegs from the mlm label of English
    This is used to enable optimizing in a multi-class classification fashion
    param: en_labels: torch.Size([batch_size, sequence_length])
    """

    def convert_en_mlm_label_to_all(self, en_labels: torch.Tensor):
        all_labels = [
            en_labels.clone() for _ in range(len(self.global_verbalizers.keys()))
        ]
        target_token_indices = torch.where(en_labels != -100)[1]
        prompted_label_words = en_labels[en_labels != -100]
        # class_labels = [0, 1, 2, ..]
        class_labels = [
            self.verbalizers[self.tokenizer.decode(prompted_label_word)]
            for prompted_label_word in prompted_label_words
        ]
        for i, lan in enumerate(self.global_verbalizers.keys()):
            for j, (target_token_index, class_label) in enumerate(
                zip(target_token_indices, class_labels)
            ):
                label_word_for_this_lan = self.inverted_global_verbalizers[lan][
                    class_label
                ]
                all_labels[i][j][target_token_index] = self.tokenizer.encode(
                    label_word_for_this_lan
                )[-2]
        return all_labels

    """
    Derive the mlm label for two languages from the mlm label of English
    This is used to enable optimizing in a multi-class classification fashion
    param: en_labels: torch.Size([batch_size, sequence_length]), augment is an index
    to indicate which language to use
    """

    def convert_en_mlm_label_to_bilingual(self, en_labels: torch.Tensor, augment: torch.Tensor):
        all_labels = [
            en_labels.clone() for _ in range(2)
        ]
        bilingual_langs = ["en"] + [self.target_langs[int(augment) - 1]]
        target_token_indices = torch.where(en_labels != -100)[1]
        prompted_label_words = en_labels[en_labels != -100]
        # class_labels = [0, 1, 2, ..]
        class_labels = [
            self.verbalizers[self.tokenizer.decode(prompted_label_word)]
            for prompted_label_word in prompted_label_words
        ]
        for i, lan in enumerate(bilingual_langs):
            for j, (target_token_index, class_label) in enumerate(
                zip(target_token_indices, class_labels)
            ):
                label_word_for_this_lan = self.inverted_global_verbalizers[lan][
                    class_label
                ]
                all_labels[i][j][target_token_index] = self.tokenizer.encode(
                    label_word_for_this_lan
                )[-2]
        return all_labels

    """
    convert the classification label to MaskedLM label for prompting
    e.g. [0] -> [-100 ... -100 TOKENIZER_ID(yes) -100 ...]
    Note that there's two hard coding
    (1) for the index of <mask> (-4), which is determined by the template
    (2) for the label word index (-1), since "_" is tokenized separately sometimes
    e.g. 
    >>> tokenizer.tokenize("न")
    ['▁n', 'न']
    >>> tokenizer.encode("nन")
    [0, 653, 998, 2]
    """

    def get_prompt_labels(self, prompts, labels):
        prompt_labels = []
        # exchange key and value of the verbalizer dict to simplify the operation below
        for i, label in enumerate(labels):
            encoded = self.tokenizer.encode(prompts[i])
            p_label = [-100] * len(encoded)
            verb = self.inverted_verbalizers[label]
            # normal prompting training data
            if self.tokenizer.mask_token_id in encoded:
                mask_idx = encoded.index(self.tokenizer.mask_token_id)
                p_label[mask_idx] = self.tokenizer.encode(verb)[-2]
            prompt_labels.append(p_label)
        return prompt_labels

    """
    Convert an NLI training example to a prompt for the LM
    """

    def convert_examples_to_prmopts(self, examples):
        if self.dataset == "xnli" or self.dataset == "paws-x":
            prompts = []
            # {premise.} {<sep_token>} {hypothesis?} {<mask_token>} .
            template = "{} {} {} {} ."
            for text_a, text_b in zip(examples[self.text_a_key], examples[self.text_b_key]):
                # append a dot in the end
                if text_a[-1] != ".":
                    if (
                        (text_a[-1] <= "z" and text_a[-1] >= "a")
                        or (text_a[-1] <= "Z" and text_a[-1] >= "A")
                        or (text_a[-1] <= "9" and text_a[-1] >= "0")
                    ):
                        text_a += " ."
                if text_b[-1] != "?":
                    if text_b[-1] == "!" or text_b[-1] == ".":
                        tmp_list = list(text_b)
                        tmp_list[-1] = "?"
                        text_b = "".join(tmp_list)
                    else:
                        text_b += " ?"
                prompts.append(
                    template.format(
                        text_a,
                        self.tokenizer.special_tokens_map["sep_token"],
                        text_b,
                        self.tokenizer.special_tokens_map["mask_token"],
                    )
                )
            return prompts
        else:
            raise ValueError("This dataset: {} is not supported".format(self.dataset))
    
    