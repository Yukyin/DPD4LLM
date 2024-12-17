import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
import numpy as np


class LLM_Analysis:
    def __init__(self, model_name, device, num_gpus, hidden_layers=80, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.hidden_layers = hidden_layers
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        kwargs = self._get_device_kwargs()
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code='chatglm' in self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code='chatglm' in self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True, config=config, **kwargs)
        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        return model, tokenizer

    def _get_device_kwargs(self):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{self.model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        return kwargs

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            print(f"Added stop word: {stop_word} with the ids {stop_word_ids}", flush=True)
            self.stopping_criteria.append(LLamaQaStoppingCriteria([stop_word_ids]))

    def lm_score(self, input_text1, input_text2, mode='baseline', **kwargs):
        with torch.no_grad():
            input_ids, prefix_ids, continue_ids = self._prepare_input_ids(input_text1, input_text2)
            if 'baseline' in mode:
                log_probs = self._calculate_log_probs(input_ids, prefix_ids, continue_ids, mode)
        return log_probs, None

    def _prepare_input_ids(self, input_text1, input_text2):
        input_text = input_text1 + input_text2
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
        continue_ids = input_ids[0, prefix_ids.shape[-1]:]
        return input_ids, prefix_ids, continue_ids

    def _calculate_log_probs(self, input_ids, prefix_ids, continue_ids, mode):
        if 'layer_wise' in mode:
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_layers = outputs[-1]
            log_probs = self._calculate_layer_wise_log_probs(hidden_layers, prefix_ids, continue_ids)
        else:
            outputs = self.model(input_ids)[0].squeeze(0)
            outputs = outputs.log_softmax(-1)  # logits to log probs
            outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
        return log_probs

    def _calculate_layer_wise_log_probs(self, hidden_layers, prefix_ids, continue_ids):
        log_probs = []
        for ind in range(1, len(hidden_layers)):
            output = hidden_layers[ind].squeeze(0)
            output = self.model.lm_head(output)
            output = output.log_softmax(-1)  # logits to log probs
            output = output[prefix_ids.shape[-1] - 1: -1, :]
            log_prob = output[range(output.shape[0]), continue_ids].sum().item()
            log_probs.append(log_prob)
        return log_probs
