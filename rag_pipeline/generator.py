# rag_pipeline/generator.py

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from rag_pipeline.utils.logger import get_logger
logger = get_logger(__name__)
import os


# hf_model_name = "tiiuae/falcon-rw-1b"

##bigscience/bloom-560m

class Generator:
    def __init__(self, hf_model_name="tiiuae/falcon-rw-1b"):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        offload_path = os.path.join(os.getcwd(), "offload")
        os.makedirs(offload_path, exist_ok=True)
        # self.model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        # if device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = device
        # self.model.to(self.device)
        # self.model = AutoModelForCausalLM.from_pretrained(
        # hf_model_name,
        # device_map="auto",  # Smartly places model on available device
        # low_cpu_mem_usage=True,
        # offload_folder="offload")  # Avoids full CPU RAM loading

        # To avoid excessive generation
        # ✅ Force full loading to CPU to avoid meta-tensor issues
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            device_map=None,               # ⛔ Don't use "auto" here — manually control device
            low_cpu_mem_usage=False,       # ✅ Force loading weights into memory
            trust_remote_code=True         # ✅ Required for Falcon
        ).to("cpu")  # ✅ Ensure it's fully loaded to CPU
        self.max_output_tokens = 256

    def build_prompt(self, query, retrieved_chunks):
        prompt = (
            "You are an assistant helping users query industrial product data.\n"
            "The user will ask questions that combine product attributes, classifications, pricing, and reviews.\n\n"
            f"User Question: {query}\n\n"
            "ONLY use the information from the following documents to answer the question.\n"
            "If information is missing, reply: 'Information not available.'\n"
            "Answer concisely using tables or bullet points where needed.\n\n"
            "Context Documents:\n"
        )
        for idx, doc in enumerate(retrieved_chunks):
            prompt += f"--- Document {idx+1} ---\n{doc['text']}\n"
        prompt += "\nYour Answer:"
        return prompt



    def generate(self, query, retrieved_chunks):
        logger.info("🧠 Building prompt and generating response...")  # ✅ log this
        prompt = self.build_prompt(query, retrieved_chunks)
        logger.info(f"📝 Prompt length: {len(prompt.split())} words")  # ✅ log prompt 
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  # Send to correct device

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_tokens,
            # do_sample=False,
            do_sample=True, 
            temperature=0.7, 
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return answer.strip()
