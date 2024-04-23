from helper_functions import load_llm, get_config

config = get_config()
model_id = config.get('Models', 'model_id')
embedding_model_id = config.get('Models', 'embedding_model_id')
temperature = config.getint('Hyperparameters', 'temperature')
max_new_tokens = config.getint('Hyperparameters', 'max_new_tokens')

llm = load_llm(model_id, embedding_model_id,temperature,max_new_tokens)
