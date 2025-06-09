from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template
from dotenv import load_dotenv

load_dotenv('../.env')

def format_to_openai_messages(prompt: str, completion: str = None) -> list:
    """Format a prompt-completion pair into a conversation list."""
    conversation = [{"role": "user", "content": prompt}]  # Use model's expected format directly'
    if completion:
        conversation.append({"role": "assistant", "content": completion})
    return conversation

def main():
    # Create a basic tokenizer just for template formatting
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
    
    # Configure with Llama chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
    )
    
    # Load dataset
    dataset = load_dataset("Trelis/touch-rugby-rules", split="train")
    
    # Take a few examples
    examples = dataset.select(range(3))
    
    print("=== Dataset Examples ===")
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {example['prompt']}")
        print(f"Completion: {example['completion']}")
        
        # Format as conversation
        conversation = format_to_openai_messages(example['prompt'], example['completion'])
        
        # Show formatted template using unsloth's template
        print("\nFormatted with Llama template:")
        formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
        print(formatted)
        
        print("\nFormatted with generation prompt:")
        formatted_with_prompt = tokenizer.apply_chat_template(
            format_to_openai_messages(example['prompt']), 
            tokenize=False,
            add_generation_prompt=True
        )
        print(formatted_with_prompt)
        print("-" * 80)

if __name__ == "__main__":
    main()