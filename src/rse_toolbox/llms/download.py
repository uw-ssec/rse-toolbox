from huggingface_hub import snapshot_download


def download_llm(model_id: str, local_dir: str | None = None) -> None:
    """
    Download the latest language model from the Hugging Face Hub.

    Parameters
    ----------
    model_id : str
        The model path on the Hugging Face Hub,
        which includes the organization and the model name.
        For example, "allenai/OLMo-7B-Instruct".
    local_dir : str, optional
        The local directory where the model will be downloaded.
    """
    if local_dir is None:
        local_dir = model_id.split("/")[-1]
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main",
    )
