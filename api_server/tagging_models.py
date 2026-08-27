import huggingface_hub


MODEL_ID = "cl-tagger.v2"
MODEL_REPO = "cella110n/cl_tagger_v2"
MODEL_REVISION = "b57909b8e9c63f71e208a26473e7aabdf45ed6b6"
MODEL_SUBDIRECTORY = "v2_00"
MODEL_FILENAME = "model.onnx"
MODEL_DATA_FILENAME = "model.onnx.data"
VOCABULARY_FILENAME = "model_vocabulary.json"


def _download_model_file(filename: str) -> str:
    return huggingface_hub.hf_hub_download(
        MODEL_REPO,
        f"{MODEL_SUBDIRECTORY}/{filename}",
        revision=MODEL_REVISION,
    )


def get_model_and_labels(model: str) -> tuple[str, str]:
    if model != MODEL_ID:
        raise ValueError(f"Model {model} is not supported. Supported model: {MODEL_ID}")

    # ONNX Runtime resolves external data relative to model.onnx, so make sure
    # the external weights are present in the same cached snapshot first.
    _download_model_file(MODEL_DATA_FILENAME)
    model_path = _download_model_file(MODEL_FILENAME)
    vocabulary_path = _download_model_file(VOCABULARY_FILENAME)
    return model_path, vocabulary_path
