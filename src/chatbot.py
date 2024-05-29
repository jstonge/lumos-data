# /bin/python
import os
import regex as re
import random
import subprocess
import time
import requests
import json

LLAMACPP_PARAMS = {
    'num_gpu_layers': 100,
    'context_window': 4096,
    'model_dir': "/users/a/c/achawla1/main/Catalogue/LLM/Models/llama_cpp/",
    'llama_cpp_dir': "/users/a/c/achawla1/main/Catalogue/LLM/llama.cpp/",
    'port': 8080,
    'num_prediction_tokens': 20,
    'batch_size': 4096
}

LLAMACPP_TIMEOUT = 1000


class Chatbot:
    def __init__(self):
        self.model
        self.temperature
        self.grammar
        self.num_prediction_tokens

    def generate_response(self, prompt):
        pass

    def generate_hash(self, return_params=False):
        """
        Generate a hash of the parameters of the chatbot.
        """
        parameters = (
            self.model,
            self.temperature,
            self.grammar,
            self.num_prediction_tokens,
        )
        if return_params:
            return hash(parameters), parameters
        else:
            return hash(parameters)

    def save_params(self, db_connector_writer,db_connector_reader, table_name="chatbot_params"):
        """
        Save the parameters of the chatbot to a SQL table and hash them to create an id.
        """

        # save the paramters of the chatbot to a sql table and hash them to create an id
        hash_id, parameters = self.generate_hash(return_params=True)
        values = [hash_id] + list(parameters)
        # check if table exists and createa it if it does not
        db_connector_writer.execute_query(
            f"CREATE TABLE IF NOT EXISTS {table_name} (id BIGINT PRIMARY KEY, model TEXT, temperature REAL, grammar TEXT, num_prediction_tokens INTEGER)"
        )
        # check if the id already exists in the table_name
        response = db_connector_reader.execute_query(f"SELECT * FROM {table_name} WHERE id = {hash_id}").all()
        if len(response) ==  0: 
            with db_connector_writer.engine.connect() as conn:
                    # Start a transaction
                        with conn.begin():
                            query = text("INSERT INTO chatbot_params VALUES (:id,:model,:temperature,:grammar,:num_prediction_tokens)")
                            values_dict = {
                                    'id':values[0],
                                    'model':values[1],
                                    'temperature':values[2],
                                    'grammar':values[3],
                                    'num_prediction_tokens':values[4]
                                    }
                            conn.execute(query,values_dict)


class LlamaCppChatbot(Chatbot):
    def __init__(self, model, temperature = None, grammar = None):

        models_avail = self.list_models()

        assert model in models_avail, f"Invalid model specification, correct options- {", ".join(models_avail)}"

        self.model_weights = f"{LLAMACPP_PARAMS['model_dir']}{model}.gguf"
        self.model = model
        self.temperature = temperature
        self.num_prediction_tokens = LLAMACPP_PARAMS["num_prediction_tokens"]
        self.context_window = LLAMACPP_PARAMS["context_window"]
        self.num_gpu_layers = LLAMACPP_PARAMS["num_gpu_layers"]
        self.batch_size = LLAMACPP_PARAMS["batch_size"]

        self.port = LLAMACPP_PARAMS["port"]
        self.endpoint = f"http://localhost:{LLAMACPP_PARAMS['port']}/completion"
        self.grammar = grammar

        if self.grammar:
            self.load_grammar()

        self.llamacpp_process = None
        self.load_model()
    
    def list_models(self):
        return [_.split('.')[0] for _ in os.listdir(LLAMACPP_PARAMS["model_dir"])]

    def load_model(self):
        """
        Load the model into the LlamaCpp server.
        """
        print("Starting Server ...................", flush=True)
        model_loaded = False
        self.llamacpp_process = subprocess.Popen(f"{LLAMACPP_PARAMS['llama_cpp_dir']}build/bin/server -m {self.model_weights} -c {self.context_window} -ngl {self.num_gpu_layers} -b {self.batch_size} --port {self.port}".split(), stdout=subprocess.PIPE)
        while not model_loaded:
            output = self.llamacpp_process.stdout.readline().decode('utf-8')
            if f'"port":{self.port}' in str(output):
                model_loaded = True

        print("Model loaded", flush=True)
    
    def load_grammar(self):
        """
        Load a grammar file for structured responses.
        """
        with open(f"{LLAMACPP_PARAMS['llama_cpp_dir']}grammars/{self.grammar}") as f:
            self.grammar_file = f.read()
    
    def structure_prompt(self, prompt_dict):
        """
        Structure the prompt in correct chat format for the given model. Also accomodates n-shot examples.
        """
        full_prompt = f'''<s>[INST] <<SYS>> {prompt_dict['system_prompt']} <</SYS>>'''

        for example_prompt in prompt_dict['example_prompts']:
            user_query_list = example_prompt.split('\n')[:-1]
            bot_answer = f'''{example_prompt.split('\n')[-1]}'''

            user_query = "\n".join(user_query_list)

            full_prompt += f'''\n{user_query}[/INST]\n{bot_answer} </s>\n<s>[INST]'''
        
        full_prompt += f'''{prompt_dict['testing_prompt']}[/INST]'''

        return full_prompt
    
    def generate_response(self, prompt_dict):
        """
        Generate a response from the LlamaCpp model.
        """
        prompt = self.structure_prompt(prompt_dict)
        data_dict = {"prompt": prompt, "n_predict": self.num_prediction_tokens}

        if self.grammar:
            data_dict["grammar"] = self.grammar_file
        
        if self.temperature:
            data_dict["temperature"] = self.temperature

        response = None

        while response is None: # this fixes the problem where llama.cpp hangs sometimes (known issue, not solved)
            try:
                response = requests.post(self.endpoint, json=data_dict, timeout=LLAMACPP_TIMEOUT)
            except:
                print("Timeout error. Retrying ...................", flush=True)
                self.reset_model()
                response = None

        completion = json.loads(response.text) # if you get an error in loading json, this means that port that the model is loaded on is occupied by something else. Try assigning different port.

        print(completion["content"], flush=True)
        return completion["content"]

    def reset_model(self):
        """
        Reset the LlamaCpp model.
        """
        self.llamacpp_process.kill()
        time.sleep(10)
        self.load_model()