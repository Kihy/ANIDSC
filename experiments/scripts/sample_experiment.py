import argparse
import json


from ANIDSC.utils.run_script import run_file


def test_dataset():
    # benign files
    yield "new", "benign_lenovo_bulb", "../datasets/test_data"

    # Attack files (subsequent runs)
    for attack in [
        "malicious_ACK_Flooding",
        "malicious_Port_Scanning",
        "malicious_Service_Detection",
    ]:
        yield "loaded", attack, "../datasets/test_data"


def get_dataset_by_name(name: str):
    funcs = {k: v for k, v in globals().items() if callable(v)}

    if name not in funcs:
        available = ", ".join(sorted(funcs.keys()))
        raise ValueError(
            f"Function '{name}' not found. Available functions: {available}"
        )

    return funcs[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run feature extraction and pipeline detection"
    )

    parser.add_argument(
        "pipeline",type=str, 
        help="Pipeline to run (e.g., basic_pipeline, graph_pipeline, cdd_test)",
    )

    parser.add_argument(
        "dataset",type=str, 
        help="name of dataset (e.g., test)",
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Pipeline Dictionary as JSON string"
    )

    args = parser.parse_args()
    config = json.loads(args.config)

    run_file(get_dataset_by_name(args.dataset), args.pipeline, config)
