import os
import re
import json
import transformers
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
import torch
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv(override = True)
access_token_read = os.getenv('access_token_read_hf')
login(token = access_token_read)

def load_llm(model_id,embedding_model_id):
    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id,model_kwargs={'device': 'cuda'})
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto')
    model.eval()
    pipe = pipeline(model=model, tokenizer=tokenizer,
        #return_full_text=True,  # langchain expects the full text
        task='text-generation',
        temperature=0.00001,
        max_new_tokens=25000,  # max number of tokens to generate in the output
        #repetition_penalty=1.1,  # without this output begins repeating
        device_map = "auto"
    )
    llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.00001})
    return llm

def get_embeddings(embedding_model_id):
    return HuggingFaceEmbeddings(model_name=embedding_model_id,model_kwargs={'device': 'cuda'})
    
def LLM_Logs(func):
    def wrapper(*args):
        today = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(f"Logs/{today}"):
            os.makedirs(f"Logs/{today}")
        inputs = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"input": args}
        output = func(*args)
        result = {"inputs": inputs,"output": output}
        file_path = os.path.join(f"Logs/{today}", "LLM_Logs.json")
        with open(file_path, "a") as file:
            file.write(json.dumps(result) + '\n')
        return output
    return wrapper

    

    



    
