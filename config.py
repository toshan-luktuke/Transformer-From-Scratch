def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "src_lang": "en",
        "tgt_lang": "fr",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path():
    config = get_config()
    return f'{config["model_folder"]}/{config["model_filename"]}{config["src_lang"]}_{config["tgt_lang"]}.pt'