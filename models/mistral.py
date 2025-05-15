# External imports
import bitsandbytes as bnb
from copy import copy
from datasets import Dataset
import json
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging, Trainer, TrainingArguments, default_data_collator
import time
import torch
from torch.nn import Linear
import os

# Internal imports
from models.model import LanguageModel

#
# Restrict max memory usage to avoid spikes
#
# Added due to out-of-memory errors during LoRA
#
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class Mistral(LanguageModel):
    #
    # Load the model and set the desired configuration parameters.
    #
    def __init__(self, config : str):
        print('Initializing Mistral', flush=True)
        #
        # Load the model configuration
        #
        with open(config, 'r') as file:
            self.config = json.load(file)
        #
        # Disable HuggingFace logging for better performance
        #
        logging.set_verbosity_error()
        #
        # Configure 4-bit quantization to fit in 8GB VRAM
        #
        #self.config['bits_and_bytes']['bnb_4bit_compute_dtype'] = eval(self.config['bits_and_bytes']['bnb_4bit_compute_dtype'])
        #self.bnb_config = BitsAndBytesConfig(
        #    **self.config['bits_and_bytes']
        #)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        #
        # DEBUG
        #
        def log_mem():
            free, total = torch.cuda.mem_get_info()
            print(f'Free: {round(free/1e9, 2)} GB; Total: {round(total/1e9, 1)} GB; Remaining: {round(free/total * 100, 2)}%', flush=True)
        print()
        print('Initial memory:', flush=True)
        log_mem()

        #
        # Load the model and tokenizer
        #
        self.name = self.config['name']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, 
            trust_remote_code=True,
        )
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        #
        # DEBUG
        #
        print()
        print('After loading the tokenizer:', flush=True)
        log_mem()

        base_model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=self.bnb_config,
            device_map='auto',
            trust_remote_code=True
        )

        #
        # DEBUG
        #
        print()
        print('After loading the model:', flush=True)
        log_mem()
        
        
        #
        # The Mistral model is quantized so we have to use a LoRA adapter.
        #
        base_model = prepare_model_for_kbit_training(base_model)

        #
        # DEBUG
        #
        print()
        print('After preparing the model:', flush=True)
        log_mem()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, lora_config)

        #
        # DEBUG
        #
        print()
        print('After peft:', flush=True)
        log_mem()
        print()
        print("Model is on:", next(self.model.parameters()).device, flush=True)
        print()
        print('Device map:', base_model.hf_device_map, flush=True)
        print()

    #
    # Given strings with the user and system prompts, query the LLM
    # and return the response.
    #
    def generate_response(self, system_prompts : list[str], user_prompts : list[str], temp: float=None) -> list[str]:
        #
        # Get prompt batch size
        #
        assert len(system_prompts) == len(user_prompts), "Prompt list length mismatch."
        N = len(system_prompts)
        #
        # Adjust temperature if it was passed as a function argument
        #
        generate_args = copy(self.config['generate'])
        if temp is not None:
            if temp != 0:
                generate_args['temperature'] = temp
            else:
                # Can't have a temperature = 0
                # Instead, we tell the llm to sample deterministicly.
                del generate_args['temperature']
                generate_args['do_sample'] = False
        #
        # Format each system+user prompt pair together
        #
        prompts = []
        for system, user in zip(system_prompts, user_prompts):
            #
            # Convert the prompt strings into the huggingface message format
            #
            message = self.prompts_to_messages(system, user)
            #
            # Format the message into Mistral's prompt format
            #
            mistral_prompt = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            #
            # Add this to the prompts list
            #
            prompts.append(mistral_prompt)
        #
        # Batch the prompts to the GPU to avoid memory overflows.
        #
        responses = []
        #
        # Token aware batching, limit the number of tokens in a batch.
        #
        MAX_TOKENS_PER_BATCH = 5000
        i = 0
        #
        # For each batch...
        #
        while i < N:
            batch = []
            tokens_in_batch = 0
            #
            # Keep loading prompts into the batch until
            # the token limit is hit.
            #
            while i < N:
                #
                # Get the token length of this prompt
                #
                prompt_token_len = len(self.tokenizer(prompts[i])['input_ids'])
                #
                # If adding this prompt to the batch exceeds the token limit,
                # then move on.
                #
                if tokens_in_batch + prompt_token_len > MAX_TOKENS_PER_BATCH:
                    break
                #
                # Else, add the prompt to this batch.
                #
                batch.append(prompts[i])
                tokens_in_batch += prompt_token_len
                i += 1
            #
            # Log
            #
            print('Tokens in batch:', tokens_in_batch, flush=True)
            #
            # Tokenize the prompts in this batch
            #
            inputs = self.tokenizer(batch,
                                    truncation=True, # Truncate prompts that exceed the model's maximum prompt length.
                                    padding=True,
                                    return_tensors='pt').to('cuda')
            #
            # Get the length of each prompt in tokens
            #
            input_ids = inputs['input_ids']
            prompt_lens = (input_ids != self.tokenizer.pad_token_id).sum(dim=1)
            #
            # Perform the query
            #
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_args,
                    pad_token_id=self.tokenizer.eos_token_id, # supresses a warning message
                )
            #
            # Decode the response from the instruction
            #
            for output, prompt_len in zip(outputs, prompt_lens):
                #
                # Slice the output to only get the tokens generated by the llm
                #
                generated_tokens = output[prompt_len:]
                #
                # Decode the generated tokens
                #
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                #
                # Save the decoded text to the responses list.
                #
                responses.append(decoded)
        #
        # Clear the GPU cache before returning
        #
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        #
        # Return the response
        #
        return responses
    
    #
    # Given data, fine-tune the model.
    #
    def train(self, data) -> None:
        # Simple tokenization loop — fast for small datasets
        tokenized_examples = []
        for ex in data:
            full = (
                "<s>[INST] <<SYS>>\n" + ex["system_prompt"] + "\n<</SYS>>\n"
                + ex["user_prompt"] + " [/INST] "
                + ex["response"] + self.tokenizer.eos_token
            )
            tokenized = self.tokenizer(
                full,
                truncation=True,
                padding='max_length',  # Important to avoid dynamic length and fragmentation
                max_length=2048,
                add_special_tokens=True,
            )
            sep_idx = tokenized["input_ids"].index(self.tokenizer.eos_token_id, 1)
            labels = tokenized["input_ids"][:]
            for i in range(sep_idx + 1):
                labels[i] = -100
            tokenized["labels"] = labels
            tokenized_examples.append(tokenized)

        # Convert to HuggingFace Dataset
        tokenized_ds = Dataset.from_list(tokenized_examples)
        
        #
        # Set training arguments
        #
        training_args = training_args = TrainingArguments(
            output_dir="./outputs",
            per_device_train_batch_size=1,     # try 1–2
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,     # effective batch size ≈ 8
            num_train_epochs=self.config['train']['num_train_epochs'],
            learning_rate=self.config['train']['learning_rate'],
            warmup_ratio=0.03,                 # ~3% of steps
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            save_steps=0,
            save_total_limit=0,
            fp16=True,                         # or bf16 if supported
            optim="adamw_torch",               # the new PyTorch optimizer
            gradient_checkpointing=True,       # save memory
            label_names=["labels"]
        )
        #
        # Train
        #
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=default_data_collator,
        )
        #
        # NOTE - During training we are hitting an occassional an illegal memory access
        #        error. I think it might be because the GPU is running out of memory
        #        mid-training. For now, I'm wrapping it in a try-except block. In the future,
        #        we should figure out how to configure the training parameters to prevent this
        #        from happening.
        #
        # Variables
        #  - MAX_TRAIN_RETRIES: Number of times to retry the train block before raising an error.
        #  - retries: Counter to control whether or not we retry the train block.
        #
        print('Starting training', flush=True)
        MAX_TRAIN_RETRIES = 2
        retries = 0
        #
        # Stop if we have retried training too many times.
        #
        while retries < MAX_TRAIN_RETRIES:
            #
            # Try-block - try to running the training function.
            #
            try:
                trainer.train()
                break # If we reach here, then no error occurred and we don't need to retry.
            #
            # Except-block - either throw an error or retry depending on error type.
            #
            except RuntimeError as e:
                msg = str(e)
                #
                # If the error is an illegal memory access, then retry training.
                #
                if 'CUDA error: an illegal memory access was encountered' in msg:
                    print("WARNING: Caught CUDA illegal memory access. Cleaning up GPU memory...", flush=True)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    retries += 1
                #
                # Raise a fatal error for any other error type.
                #
                else:
                    raise SystemExit(msg)
        print('Finished training', flush=True)
        #
        # Error case - we exit the loop after too many retries
        #
        # Optionally, save model checkpoint here
        if retries == MAX_TRAIN_RETRIES:
            raise SystemExit("Fatal CUDA error during training. Please restart the process.")

    #
    # Save the model to disk
    #
    def save(self):
        save_dir = self.config['save_dir']
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)