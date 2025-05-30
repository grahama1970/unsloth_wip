import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from loguru import logger

# Configuration
MODEL_NAME = "unsloth/Phi-3.5-mini-instruct"  # Base model
ADAPTER_PATH = "/home/grahama/dev/vllm_lora/training_output/Phi-3.5-mini-instruct_touch-rugby-rules_adapter/final_model"  # Adapter path


def setup_model_and_tokenizer(model_name: str, adapter_path: str):
    """
    Load the model and tokenizer and apply the adapter.

    Args:
        model_name (str): The name of the base model.
        adapter_path (str): The path to the trained LoRA adapter.

    Returns:
        Tuple: Configured model and tokenizer.
    """
    logger.info("Loading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True  # Ensure compatibility with LoRA adapter training
    )

    logger.info("Loading adapter...")
    model.load_adapter(adapter_path)

    logger.info("Configuring tokenizer with Phi-3 chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",
        mapping={
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt"
        }
    )

    logger.info("Preparing model for inference...")
    FastLanguageModel.for_inference(model)  # Prepare the model for Unsloth inference

    return model, tokenizer


def generate_response(model, tokenizer, question: str):
    """
    Generate a response for a given question.

    Args:
        model: The configured model.
        tokenizer: The configured tokenizer.
        question (str): The input question.

    Returns:
        str: Generated response.
    """
    logger.info("Formatting input with chat template...")
    input_conversation = [{"role": "user", "content": question}]
    formatted_prompt = tokenizer.apply_chat_template(
        input_conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Tokenizing input...")
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    logger.info("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, ADAPTER_PATH)

    questions = [
        "What is a touchdown in Touch Rugby?",
        "How many players are on a Touch Rugby team?",
        "What happens after a touchdown in Touch Rugby?"
    ]

    for question in questions:
        logger.info(f"Q: {question}")
        response = generate_response(model, tokenizer, question)
        logger.info(f"A: {response}")
        logger.info("-" * 80)


if __name__ == "__main__":
    main()
