
import yaml

def load_yaml_as_dict(node, yaml_param_name):
    filepath = node.get_parameter(yaml_param_name).get_parameter_value().string_value
    with open(filepath, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)