import os
from typing import Optional


def _get_data_dir(data_dir: Optional[str]) -> str:
    if data_dir:
        return data_dir
    env = os.environ.get("RAPO_DATA_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _ensure_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} at: {path}")


def _require(module: str, install_hint: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Missing dependency `{module}`. Install {install_hint}.") from e


def load_data(dataset_name: str, max_item: int = 1000, data_dir: Optional[str] = None):
    data_dir = _get_data_dir(data_dir)

    if dataset_name == "stratasword":
        pd = _require("pandas", "`pip install pandas`")
        records = []
        for level in (1, 2, 3):
            csv_path = os.path.join(data_dir, f"strata_sword_en_level_{level}.csv")
            _ensure_exists(csv_path, f"stratasword level-{level} csv")
            df = pd.read_csv(csv_path)
            df["level"] = level
            df = df.rename(columns={"jailbreak instruction": "prompt"})
            df["harmful_label"] = 1
            records.extend(df.to_dict("records"))
        return records[:max_item]

    if dataset_name == "sorrybench":
        pd = _require("pandas", "`pip install pandas`")
        csv_path = os.path.join(data_dir, "sorrybench.csv")
        _ensure_exists(csv_path, "sorrybench csv")
        df = pd.read_csv(csv_path)
        if "prompt_style" in df.columns:
            allowed = [
                "base",
                "authority_endorsement",
                "evidence-based_persuasion",
                "expert_endorsement",
                "logical_appeal",
                "role_play",
            ]
            df = df[df["prompt_style"].isin(allowed)]
        df["harmful_label"] = 1
        return df.to_dict("records")[:max_item]

    if dataset_name == "jailbreakbench":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "JailbreakBench")
        _ensure_exists(path, "JailbreakBench dataset folder")
        df = load_dataset(path)["train"].to_pandas()
        if "subset" in df.columns:
            df = df[df["subset"] == "harmful"]
        df["harmful_label"] = 1
        return df.to_dict("records")[:max_item]

    if dataset_name == "wildjailbreak":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "wildjailbreak")
        _ensure_exists(path, "wildjailbreak dataset folder")
        df = load_dataset(path, "eval", delimiter="\t", keep_default_na=False)["train"].to_pandas()
        if "label" in df.columns:
            df = df[df["label"] == 1]
        if "adversarial" in df.columns:
            df["prompt"] = df["adversarial"]
        df["harmful_label"] = 1
        return df.to_dict("records")[:max_item]

    if dataset_name == "strongreject":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "StrongReject")
        _ensure_exists(path, "StrongReject dataset folder")
        df = load_dataset(path)["train"].to_pandas()
        df["harmful_label"] = 1
        return df.to_dict("records")[:max_item]

    if dataset_name == "xstest":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "XsTest")
        _ensure_exists(path, "XsTest dataset folder")
        df = load_dataset(path)["test"].to_pandas()
        if "label" in df.columns:
            df = df[df["label"] == "safe"]
        df["harmful_label"] = 0
        return df.to_dict("records")[:max_item]

    if dataset_name == "starbenign":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "STAR-benign")
        _ensure_exists(path, "STAR-benign dataset folder")
        df = load_dataset(path)["train"].to_pandas()
        if "question" in df.columns:
            df["prompt"] = df["question"]
        df["harmful_label"] = 0
        return df.to_dict("records")[:max_item]

    if dataset_name == "star":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "STAR-1K")
        _ensure_exists(path, "STAR-1K dataset folder")
        df = load_dataset(path)["train"].to_pandas()
        if "question" in df.columns:
            df["prompt"] = df["question"]
        df["harmful_label"] = 1
        return df.to_dict("records")[:max_item]

    if dataset_name == "harmbench":
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset
        path = os.path.join(data_dir, "Harmbench")
        _ensure_exists(path, "Harmbench dataset folder")
        df = load_dataset(path, "standard")["train"].to_pandas()
        df["harmful_label"] = 1
        return df.to_dict("records")[:max_item]

    raise NotImplementedError(f"Unknown dataset_name: {dataset_name}")
