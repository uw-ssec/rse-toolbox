from enum import IntEnum

import llama_cpp


# https://github.com/ggerganov/llama.cpp/blob/bc4bba364fb96d908f2698e908648df5e6f55e02/llama.h#L109-L145
class llama_ftype(IntEnum):
    LLAMA_FTYPE_ALL_F32 = 0
    LLAMA_FTYPE_MOSTLY_F16 = 1
    LLAMA_FTYPE_MOSTLY_Q4_0 = 2
    LLAMA_FTYPE_MOSTLY_Q4_1 = 3
    LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    # LLAMA_FTYPE_MOSTLY_Q4_2 = 5 # support has been removed
    # LLAMA_FTYPE_MOSTLY_Q4_3 = 6 # support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    LLAMA_FTYPE_MOSTLY_Q5_0 = 8
    LLAMA_FTYPE_MOSTLY_Q5_1 = 9
    LLAMA_FTYPE_MOSTLY_Q2_K = 10
    LLAMA_FTYPE_MOSTLY_Q3_K_S = 11
    LLAMA_FTYPE_MOSTLY_Q3_K_M = 12
    LLAMA_FTYPE_MOSTLY_Q3_K_L = 13
    LLAMA_FTYPE_MOSTLY_Q4_K_S = 14
    LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
    LLAMA_FTYPE_MOSTLY_Q5_K_S = 16
    LLAMA_FTYPE_MOSTLY_Q5_K_M = 17
    LLAMA_FTYPE_MOSTLY_Q6_K = 18
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19
    LLAMA_FTYPE_MOSTLY_IQ2_XS = 20
    LLAMA_FTYPE_MOSTLY_Q2_K_S = 21
    LLAMA_FTYPE_MOSTLY_IQ3_XS = 22
    LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23
    LLAMA_FTYPE_MOSTLY_IQ1_S = 24
    LLAMA_FTYPE_MOSTLY_IQ4_NL = 25
    LLAMA_FTYPE_MOSTLY_IQ3_S = 26
    LLAMA_FTYPE_MOSTLY_IQ3_M = 27
    LLAMA_FTYPE_MOSTLY_IQ2_S = 28
    LLAMA_FTYPE_MOSTLY_IQ2_M = 29
    LLAMA_FTYPE_MOSTLY_IQ4_XS = 30
    LLAMA_FTYPE_MOSTLY_IQ1_M = 31
    LLAMA_FTYPE_MOSTLY_BF16 = 32
    LLAMA_FTYPE_GUESSED = 1024  # not specified in the model file


class QuantizationMethods(IntEnum):
    F32 = llama_ftype.LLAMA_FTYPE_ALL_F32
    F16 = llama_ftype.LLAMA_FTYPE_MOSTLY_F16
    Q4_0 = llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_0
    Q4_1 = llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_1
    Q8_0 = llama_ftype.LLAMA_FTYPE_MOSTLY_Q8_0
    Q5_0 = llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_0
    Q5_1 = llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_1
    Q2_K = llama_ftype.LLAMA_FTYPE_MOSTLY_Q2_K
    Q2_K_S = llama_ftype.LLAMA_FTYPE_MOSTLY_Q2_K_S
    Q3_K_S = llama_ftype.LLAMA_FTYPE_MOSTLY_Q3_K_S
    Q3_K_M = llama_ftype.LLAMA_FTYPE_MOSTLY_Q3_K_M
    Q3_K_L = llama_ftype.LLAMA_FTYPE_MOSTLY_Q3_K_L
    Q4_K = llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_K_M  # alias
    Q4_K_S = llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_K_S
    Q4_K_M = llama_ftype.LLAMA_FTYPE_MOSTLY_Q4_K_M
    Q5_K = llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_K_M  # alias
    Q5_K_S = llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_K_S
    Q5_K_M = llama_ftype.LLAMA_FTYPE_MOSTLY_Q5_K_M
    Q6_K = llama_ftype.LLAMA_FTYPE_MOSTLY_Q6_K
    IQ2_XXS = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_XXS
    IQ2_XS = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_XS
    IQ3_XS = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_XS
    IQ3_XXS = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_XXS
    IQ1_S = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ1_S
    IQ4_NL = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ4_NL
    IQ3_S = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_S
    IQ3_M = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ3_M
    IQ2_S = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_S
    IQ2_M = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ2_M
    IQ4_XS = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ4_XS
    IQ1_M = llama_ftype.LLAMA_FTYPE_MOSTLY_IQ1_M
    BF16 = llama_ftype.LLAMA_FTYPE_MOSTLY_BF16


def quantize_gguf_model(
    input_model_file: str, output_model_file: str, method: QuantizationMethods
) -> None:
    """
    Quantize GGUF format model with specified method
    using the llama.cpp library.

    Parameters
    ----------
    input_model_file : str
        The path to the input gguf model file
    output_model_file : str
        The path to the output quantized model file
    method : QuantizationMethods
        The quantization method to use

    Raises
    ------
    RuntimeError
        If the quantization process fails
    """
    input_model_file = input_model_file.encode("utf-8")
    output_model_file = output_model_file.encode("utf-8")
    args = llama_cpp.llama_model_quantize_default_params()
    args.ftype = QuantizationMethods[method].value
    return_code = llama_cpp.llama_model_quantize(
        input_model_file, output_model_file, args
    )
    if return_code != 0:
        raise RuntimeError("Failed to quantize model")
