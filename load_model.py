import json
import os

class vllm_adapter:
    def __init__(self, model_path):
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            from transformers import AutoTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependencies. Install `vllm transformers` before using `vllm_adapter`."
            ) from e

        self.path = model_path

        self.eval_sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            skip_special_tokens=False,
        )

        model_path = self.path

        if os.path.exists(model_path + '/adapter_config.json'):
            with open(model_path + '/adapter_config.json', 'r') as f:
                lora_adapter_config = json.load(f)
                base_model_path = lora_adapter_config['base_model_name_or_path']
                lora_adapter_path = model_path
                self.use_lora = True
        else:
            self.use_lora = False
            base_model_path = model_path

        if self.use_lora:
            self.request_lora = LoRARequest(
                lora_name="lora_adapter",
                lora_int_id=1,
                lora_local_path=lora_adapter_path
            )
            self.model = LLM(
                model=base_model_path,
                enable_lora=True,   
                gpu_memory_utilization=0.4,  
                dtype="auto", 
            )
            self.tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
        else:
            self.model = LLM(
                model=base_model_path,
                gpu_memory_utilization=0.4,  
                dtype="auto", 
            )   
                
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    def generate(self, inputs, sampling_params=None):
        if sampling_params is None:
            sampling_params = self.eval_sampling_params
            
        if self.use_lora:
            outputs = self.model.generate(
                inputs,
                sampling_params,
                lora_request=self.request_lora 
            )
        else:
            outputs = self.model.generate(
                inputs,
                sampling_params,
            )
        return outputs
