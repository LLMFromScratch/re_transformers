from transformers import AutoTokenizer, PreTrainedTokenizer, Qwen3ForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

from re_transformers import Qwen3ForCausalLM as re_Qwen3ForCausalLM


MODEL_NAME = "Qwen/Qwen3-0.6B"
GENERATION_KWARGS = dict(
    max_new_tokens=10,
    num_beams=1,
    do_sample=False,
)
DECODE_KWARGS = dict(
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)


def main():
    prompt = "Hey, are you conscious? Can you talk to me?"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs: BatchEncoding = tokenizer(prompt, return_tensors="pt")

    model = Qwen3ForCausalLM.from_pretrained(MODEL_NAME)
    re_model = re_Qwen3ForCausalLM.from_pretrained(MODEL_NAME)

    # Generate with the original model.
    model_generate_ids = model.generate(inputs.input_ids, **GENERATION_KWARGS)
    model_generate_tokens = tokenizer.batch_decode(
        model_generate_ids, **DECODE_KWARGS)[0]
    print(f"{model_generate_tokens=}")

    # Generate with the re-implemented model.
    re_model_generate_ids = re_model.generate(
        inputs.input_ids, **GENERATION_KWARGS)
    re_model_generate_tokens = tokenizer.batch_decode(
        re_model_generate_ids, **DECODE_KWARGS)[0]
    print(f"{re_model_generate_tokens=}")


if __name__ == "__main__":
    main()
