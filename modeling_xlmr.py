from torch._C import Value
import torch.nn as nn
import torch
import os
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss

import logging
import random

logger = logging.getLogger(__name__)


class PromptXLMR(nn.Module):
    def __init__(
        self,
        model_args,
        config,
        prompt_helper,
        use_soft_prompt: bool,
        prompt_length: int,
        tune_LM: bool,
        multi_lingual_optim: bool,
        multi_lingual_label_word: bool,
        mixup_strategy: str,
        mixup_alpha: float,
    ):
        super(PromptXLMR, self).__init__()
        self.prompt_helper = prompt_helper
        self.config = config
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        self.use_soft_prompt = use_soft_prompt
        self.prompt_length = prompt_length
        self.multi_lingual_optim = multi_lingual_optim
        self.multi_lingual_label_word = multi_lingual_label_word
        self.mixup_strategy = mixup_strategy
        self.mixup_alpha = mixup_alpha
        if use_soft_prompt:
            if not tune_LM:
                for param in self.mlm_model.parameters():
                    param.requires_grad = False
            self.soft_prompt = self._init_soft_prompt(prompt_length)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        augment=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.use_soft_prompt:
            inputs_embeds = self._concat_soft_prompt_to_inputs(input_ids)
            attention_mask, labels = self._extend_accordingly(attention_mask, labels)
            # set input_ids to None to disable its functionality in forwarding
            # also enabling input_embeds. refer to the api of modeling_roberta.py
            input_ids = None

        if self.training and self.mixup_strategy == "input_embedding":
            raise NotImplementedError(
                "Input embedding mixup is not implemented yet! How to deal with the position of the <mask> token?"
            )
            inputs_embeds = self.mixup_input_embedding(input_ids)
            attention_mask, labels = self._extend_accordingly_mixup(
                attention_mask, labels
            )
            input_ids = None

        outputs = self.mlm_model.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.training and self.mixup_strategy == "hidden":
            sequence_output = self.mixup_input(input_ids, sequence_output)
        prediction_scores = self.mlm_model.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.multi_lingual_optim:
                if self.training and self.mixup_strategy == "hidden":
                    masked_lm_loss = self.mixup_loss_multilingual(
                        prediction_scores, labels, loss_fct
                    )
                else:
                    all_lan_labels = self.prompt_helper.convert_en_mlm_label_to_all(
                        labels
                    )
                    # supposing the second label is the target language
                    masked_lm_loss = torch.cat(
                        [
                            loss_fct(
                                prediction_scores.view(-1, self.config.vocab_size),
                                labels.view(-1),
                            ).unsqueeze(0)
                            for labels in all_lan_labels
                        ]
                    )
                    masked_lm_loss = torch.mean(masked_lm_loss)
            else:
                if self.training and self.mixup_strategy == "hidden":
                    masked_lm_loss = self.mixup_loss_monolingual(
                        prediction_scores, labels, loss_fct
                    )
                else:
                    masked_lm_loss = loss_fct(
                        prediction_scores.view(-1, self.config.vocab_size),
                        labels.view(-1),
                    )
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        if self.multi_lingual_label_word:
            prediction_scores = self.prompt_helper.convert_mlm_prediction_scores_to_seqcls_maximum(
                prediction_scores, labels
            )
        else:
            prediction_scores = self.prompt_helper.convert_mlm_prediction_scores_to_seqcls(
                prediction_scores, labels
            )

        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # return MaskedLMOutput(
        #     loss=masked_lm_loss,
        #     logits=prediction_scores,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

    def load_state(self, state_dict_path):
        state_dict = torch.load(
            os.path.join(state_dict_path, "pytorch_model.bin"), map_location="cpu"
        )

        load_result = self.load_state_dict(state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            logger.warn(
                f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
            )
        if len(load_result.unexpected_keys) != 0:
            logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def _init_soft_prompt(self, prompt_length: int):
        word_embedding_weights = (
            self.mlm_model.roberta.embeddings.word_embeddings.weight
        )
        sampled_indices = random.sample(
            list(range(word_embedding_weights.shape[0])), prompt_length
        )
        initialized_prompt = nn.parameter.Parameter(
            word_embedding_weights[sampled_indices].clone().detach()
        )
        return initialized_prompt

    def _concat_soft_prompt_to_inputs(self, input_ids):
        input_embeds = self.mlm_model.roberta.embeddings.word_embeddings(input_ids)
        soft_prompt_embeds = self.soft_prompt.repeat(input_ids.shape[0], 1, 1)
        input_embeds = torch.cat(
            [input_embeds[:, 0].unsqueeze(1), soft_prompt_embeds, input_embeds[:, 1:]],
            dim=1,
        )
        return input_embeds

    def _extend_accordingly(self, attention_mask: torch.Tensor, labels: torch.Tensor):
        batch_size = attention_mask.shape[0]
        prompt_attention_mask = torch.full((batch_size, self.prompt_length), 1).to(
            attention_mask.device
        )
        prompt_labels = torch.full((batch_size, self.prompt_length), -100).to(
            labels.device
        )
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        labels = torch.cat([prompt_labels, labels], dim=1)
        return attention_mask, labels

    def mixup_input(self, input_ids, sequence_output):
        Beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        assert len(input_ids) == 2, "We are supposing the batch size to be 2"
        mask_pos1, mask_pos2 = torch.where(
            input_ids == self.prompt_helper.tokenizer.mask_token_id
        )[1]
        self.lam = Beta.sample()  # would be used in mixup_loss as well
        fake_encoding = sequence_output[0].clone()
        fake_encoding[mask_pos1] = (
            self.lam * sequence_output[0][mask_pos1]
            + (1 - self.lam) * sequence_output[1][mask_pos2]
        )
        new_sequence_output = torch.cat(
            [sequence_output, fake_encoding.unsqueeze(0)], dim=0
        )
        return new_sequence_output

    def mixup_loss_multilingual(self, prediction_scores, labels, loss_fct):
        all_lan_labels = self.prompt_helper.convert_en_mlm_label_to_all(labels)
        masked_lm_loss_normal = torch.cat(
            [
                loss_fct(
                    prediction_scores[:2].view(-1, self.config.vocab_size),
                    labels.view(-1),
                ).unsqueeze(0)
                for labels in all_lan_labels
            ]
        )
        masked_lm_loss_normal = torch.mean(masked_lm_loss_normal)
        masked_lm_loss_fake1 = torch.cat(
            [
                loss_fct(
                    prediction_scores[2].view(-1, self.config.vocab_size),
                    labels[0].view(-1),
                ).unsqueeze(0)
                for labels in all_lan_labels
            ]
        )
        mask_pos1, mask_pos2 = torch.where(labels != -100)[1]
        fake_labels_second_compnent = [
            lan_label[0].clone() for lan_label in all_lan_labels
        ]
        for i in range(len(all_lan_labels)):
            fake_labels_second_compnent[i][mask_pos1] = all_lan_labels[i][1][mask_pos2]
        masked_lm_loss_fake2 = torch.cat(
            [
                loss_fct(
                    prediction_scores[2].view(-1, self.config.vocab_size),
                    label.view(-1),
                ).unsqueeze(0)
                for label in fake_labels_second_compnent
            ]
        )
        fake_loss = (
            self.lam * masked_lm_loss_fake1 + (1 - self.lam) * masked_lm_loss_fake2
        )
        fake_loss = torch.mean(fake_loss)
        # to adjust the actual average
        overall_loss = masked_lm_loss_normal * 2 / 3 + fake_loss / 3
        return overall_loss

    def mixup_loss_monolingual(self, prediction_scores, labels, loss_fct):
        masked_lm_loss_normal = loss_fct(
            prediction_scores[:2].view(-1, self.config.vocab_size), labels.view(-1),
        )

        masked_lm_loss_fake1 = loss_fct(
            prediction_scores[2].view(-1, self.config.vocab_size), labels[0].view(-1),
        )
        mask_pos1, mask_pos2 = torch.where(labels != -100)[1]
        fake_labels_second_compnent = labels[0].clone()
        fake_labels_second_compnent[mask_pos1] = labels[1][mask_pos2]
        masked_lm_loss_fake2 = loss_fct(
            prediction_scores[2].view(-1, self.config.vocab_size),
            fake_labels_second_compnent.view(-1),
        )
        fake_loss = (
            self.lam * masked_lm_loss_fake1 + (1 - self.lam) * masked_lm_loss_fake2
        )
        # to adjust the actual average
        overall_loss = masked_lm_loss_normal * 2 / 3 + fake_loss / 3
        return overall_loss

    def mixup_input_embedding(self, input_ids):
        input_embeds = self.mlm_model.roberta.embeddings.word_embeddings(input_ids)
        Beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        assert len(input_ids) == 2, "We are supposing the batch size to be 2"
        self.lam = Beta.sample()  # would be used in mixup_loss as well
        fake_encoding = input_embeds[0].clone()
        fake_encoding = self.lam * input_embeds[0] + (1 - self.lam) * input_embeds[1]
        new_input_embeds = torch.cat([input_embeds, fake_encoding.unsqueeze(0)], dim=0)
        return new_input_embeds

    # Have a problem here, not implemented for now
    def _extend_accordingly_mixup(
        self, attention_mask: torch.Tensor, labels: torch.Tensor
    ):
        batch_size = attention_mask.shape[0]
        pass
        # attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        # labels = torch.cat([prompt_labels, labels], dim=1)
        # return attention_mask, labels
