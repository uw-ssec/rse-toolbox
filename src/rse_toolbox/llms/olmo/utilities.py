from typing import Any, Dict, List


def create_prompt_with_olmo_chat_format(
    messages: List[Dict[str, Any]],
    bos="|||IP_ADDRESS|||",
    eos="|||IP_ADDRESS|||",
    add_bos=True,
) -> str:
    """
    Create a prompt with OLMo chat format.

    Parameters
    ----------
    messages : list of dict
        List of messages with the following format:
        {
            "role": "system" | "user" | "assistant",
            "content": str
        }
    bos : str, optional
        The beginning of the string, by default "|||IP_ADDRESS|||"
    eos : str, optional
        The end of the string, by default "|||IP_ADDRESS|||"
    add_bos : bool, optional
        Whether to add the bos token to the beginning of the string, by default True

    Returns
    -------
    str
        The formatted text.

    Raises
    ------
    ValueError
        If the role is not "system", "user" or "assistant".
    """
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += (
                "<|assistant|>\n" + message["content"].strip() + eos + "\n"
            )
        else:
            raise ValueError(
                "Olmo chat template only supports 'system', 'user' and 'assistant' roles."
                f"Invalid role: {message['role']}."
            )
    formatted_text += "<|assistant|>\n"
    if add_bos:
        formatted_text = bos + formatted_text
    return formatted_text
