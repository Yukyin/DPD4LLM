import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Layer_Analyzer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_layer_outputs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  
        return hidden_states

    def get_layer_weights(self):
        layer_weights = []
        for layer in self.model.transformer.h:
            layer_weights.append({
                "attention": {
                    "query": layer.attention.query.weight,
                    "key": layer.attention.key.weight,
                    "value": layer.attention.value.weight
                },
                "dense": layer.attention.output.dense.weight,
                "intermediate": layer.intermediate.dense.weight,
                "output": layer.output.dense.weight
            })
        return layer_weights

