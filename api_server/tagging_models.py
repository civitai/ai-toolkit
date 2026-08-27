import huggingface_hub


MODEL_ID = "wd14-vit.v1"
WD14_MODEL_ID = MODEL_ID
CL_TAGGER_MODEL_ID = "cl-tagger.v2"

MODEL_FILENAME = "model.onnx"
WD14_LABELS_FILENAME = "selected_tags.csv"

CL_TAGGER_REPO = "cella110n/cl_tagger_v2"
CL_TAGGER_REVISION = "b57909b8e9c63f71e208a26473e7aabdf45ed6b6"
CL_TAGGER_SUBDIRECTORY = "v2_00"
CL_TAGGER_DATA_FILENAME = "model.onnx.data"
CL_TAGGER_VOCABULARY_FILENAME = "model_vocabulary.json"


def _download(repo: str, filename: str, revision: str) -> str:
    return huggingface_hub.hf_hub_download(
        repo,
        filename,
        revision=revision,
    )


def _download_cl_tagger_model() -> str:
    # Download external data first so ONNX Runtime can resolve it beside the graph.
    _download(
        CL_TAGGER_REPO,
        f"{CL_TAGGER_SUBDIRECTORY}/{CL_TAGGER_DATA_FILENAME}",
        CL_TAGGER_REVISION,
    )
    return _download(
        CL_TAGGER_REPO,
        f"{CL_TAGGER_SUBDIRECTORY}/{MODEL_FILENAME}",
        CL_TAGGER_REVISION,
    )


SUPPORTED_MODELS = {
    "wd14-vit-v1": {
        "model": lambda: _download(
            "SmilingWolf/wd-v1-4-vit-tagger",
            MODEL_FILENAME,
            "213a7bd66d93407911b8217e806a95edc3593eed",
        ),
        "tags": lambda: _download(
            "SmilingWolf/wd-v1-4-vit-tagger",
            WD14_LABELS_FILENAME,
            "213a7bd66d93407911b8217e806a95edc3593eed",
        ),
    },
    "wd14-vit-v2": {
        "model": lambda: _download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2",
            MODEL_FILENAME,
            "1f3f3e8ae769634e31e1ef696df11ec37493e4f2",
        ),
        "tags": lambda: _download(
            "SmilingWolf/wd-v1-4-vit-tagger-v2",
            WD14_LABELS_FILENAME,
            "1f3f3e8ae769634e31e1ef696df11ec37493e4f2",
        ),
    },
    # v1 & v2 are both using the same v2 model.
    "wd14-swinv2-v1": {
        "model": lambda: _download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            MODEL_FILENAME,
            "cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
        "tags": lambda: _download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            WD14_LABELS_FILENAME,
            "cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
    },
    "wd14-swinv2-v2": {
        "model": lambda: _download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            MODEL_FILENAME,
            "cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
        "tags": lambda: _download(
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            WD14_LABELS_FILENAME,
            "cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        ),
    },
    "cl-tagger-v2": {
        "model": _download_cl_tagger_model,
        "tags": lambda: _download(
            CL_TAGGER_REPO,
            f"{CL_TAGGER_SUBDIRECTORY}/{CL_TAGGER_VOCABULARY_FILENAME}",
            CL_TAGGER_REVISION,
        ),
    },
}


MODEL_ALIASES = {
    WD14_MODEL_ID: "wd14-vit-v1",
    "wd14-vit.v2": "wd14-vit-v2",
    CL_TAGGER_MODEL_ID: "cl-tagger-v2",
}


def get_model_and_labels(model: str):
    resolved_model = MODEL_ALIASES.get(model, model)
    if resolved_model not in SUPPORTED_MODELS:
        supported_models = sorted(set(SUPPORTED_MODELS) | set(MODEL_ALIASES))
        raise ValueError(
            f"Model {model} is not supported. Supported models are: {supported_models}"
        )

    model_config = SUPPORTED_MODELS[resolved_model]
    return model_config["model"](), model_config["tags"]()
