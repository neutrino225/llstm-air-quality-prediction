import yaml


# Load parameters from params.yml
def load_params(config_path="params.yml"):
    with open(config_path, "r") as file:
        params = yaml.safe_load(file)
    return params
