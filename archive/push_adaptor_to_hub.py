from loguru import logger
from typing import Any
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, ModelCard, ModelCardData
from dotenv import load_dotenv
from unsloth_training import TrainingConfig

load_dotenv()

def create_model_card(hub_model_id: str, template_path: str, **kwargs) -> str:
    """
    Create a model card from a markdown template file.
    
    Args:
        hub_model_id: The Hugging Face model ID
        template_path: Path to the markdown template file
        **kwargs: Additional variables to format the template with
    """
    try:
        with open(template_path, 'r') as f:
            template = f.read()
            
        # Combine provided kwargs with default values
        format_vars = {
            'hub_model_id': hub_model_id,
            'base_model': "Meta-Llama-3.1-8B-Instruct",
            **kwargs
        }
        
        # Format the template with the variables
        return template.format(**format_vars)
        
    except FileNotFoundError:
        logger.warning(f"Template file not found at {template_path}, using default model card")
        return create_default_model_card(hub_model_id, **kwargs)
    except Exception as e:
        logger.error(f"Error creating model card from template: {e}")
        return create_default_model_card(hub_model_id, **kwargs)

def create_default_model_card(hub_model_id: str, **kwargs) -> str:
    """
    Creates a basic model card when the template is not available.
    
    Args:
        hub_model_id: The Hugging Face model ID
        **kwargs: Additional metadata about the model
    """
    return f"""# {hub_model_id}

This is a LoRA adapter trained on the Meta-Llama-3.1-8B-Instruct base model.

## Model Details
- Base Model: Meta-Llama-3.1-8B-Instruct
- Adapter Type: LoRA
- Repository: {hub_model_id}
- Private: {kwargs.get('private', True)}

## Usage
This adapter should be used with the Meta-Llama-3.1-8B-Instruct base model.

## Training
This model was trained using the unsloth library.
"""

def push_adapter_to_hub(adapter_path: str, hub_model_id: str, token: str, private: bool = True, model_card_template: str = None) -> None:
    try:
        # Validate inputs
        if not token:
            raise ValueError("HuggingFace token cannot be empty")
        
        if not adapter_path or not os.path.exists(adapter_path):
            raise ValueError(f"Adapter path does not exist: {adapter_path}")
            
        if not hub_model_id or "/" not in hub_model_id:
            raise ValueError("hub_model_id must be in format 'username/model-name'")
        
        logger.info(f"Pushing adapter from {adapter_path} to HuggingFace Hub: {hub_model_id}")
        
        api = HfApi(token=token)
        
        # Create model card using the official API
        card_data = ModelCardData(
            language="en",
            license="mit",
            library_name="unsloth",
            model_name=hub_model_id
        )
        
        if model_card_template and os.path.exists(model_card_template):
            # Use template if provided
            card = ModelCard.from_template(
                card_data,
                template_path=model_card_template,
                model_id=hub_model_id,
                model_description="LoRA adapter trained on Meta-Llama-3.1-8B-Instruct base model",
            )
        else:
            # Use default template
            card = ModelCard.from_template(
                card_data,
                model_id=hub_model_id,
                model_description="LoRA adapter trained on Meta-Llama-3.1-8B-Instruct base model",
            )
        
        # Push the model card directly to the hub
        card.push_to_hub(hub_model_id)
        logger.info("Pushed model card to hub")
        
        # Always write the README.md, whether from template or default
        readme_path = os.path.join(adapter_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        logger.info("Created README.md")
        
        # List actual files in the directory
        existing_files = os.listdir(adapter_path)
        logger.info(f"Files found in adapter directory: {existing_files}")
        
        # Updated required files to include safetensors format
        required_files = [
            ["adapter_config.json", "adapter_model.safetensors"],  # Safetensors format
            ["adapter_config.json", "adapter_model.bin"],          # Binary format
            ["adapter_config.json", "pytorch_model.bin"],          # PEFT format
        ]
        
        files_found = False
        for file_set in required_files:
            if all(f in existing_files for f in file_set):
                files_found = True
                logger.info(f"Found adapter files matching format: {file_set}")
                break
                
        if not files_found:
            raise ValueError(
                f"No complete set of adapter files found in {adapter_path}. "
                f"Directory contains: {existing_files}"
            )
        
        try:
            create_repo(
                hub_model_id,
                private=private,
                token=token,
                repo_type="model",
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repository creation warning (may already exist): {e}")
        
        # Upload with ignore patterns for common unwanted files
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=hub_model_id,
            repo_type="model",
            token=token,
            ignore_patterns=["*.pyc", "__pycache__", ".git*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"]
        )
        
        logger.success(f"Successfully pushed adapter to {hub_model_id}")
        
        # Verify the upload by checking the repository
        try:
            repo_files = api.list_repo_files(hub_model_id, repo_type="model")
            required_uploaded_files = ["adapter_config.json", "adapter_model.safetensors"]
            missing_files = [f for f in required_uploaded_files if not any(f in rf for rf in repo_files)]
            if missing_files:
                logger.warning(f"Upload may be incomplete. Missing files: {missing_files}")
        except Exception as e:
            logger.warning(f"Could not verify upload: {e}")
        
    except Exception as e:
        logger.error(f"Failed to push adapter to Hub: {e}")
        raise

def save_model(model: Any, config: TrainingConfig) -> None:
    try:
        if config.save_adapter:
            adapter_path = f"{config.output_dir}/final_model"
            logger.info(f"Saving LoRA adapter to {adapter_path}")
            model.save_pretrained(adapter_path, "lora_adapter")
            logger.success("LoRA adapter saved successfully")
            
            if config.push_to_hub:
                if not config.hub_model_id:
                    raise ValueError("hub_model_id must be set to push to HuggingFace Hub")
                
                token = config.hub_token or os.getenv("HF_TOKEN")
                if not token:
                    raise ValueError("HuggingFace token not found. Set hub_token or HF_TOKEN environment variable")
                
                push_adapter_to_hub(
                    adapter_path=adapter_path,
                    hub_model_id=config.hub_model_id,
                    token=token,
                    private=config.hub_private
                )
        
        if config.save_merged:
            merged_path = config.merged_model_path
            logger.info(f"Saving merged model to {merged_path}")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            logger.success("Merged model saved successfully")
            
            if config.push_to_hub:
                merged_hub_id = f"{config.hub_model_id}-merged"
                logger.info(f"Pushing merged model to HuggingFace Hub: {merged_hub_id}")
                
                merged_model.push_to_hub(
                    merged_hub_id,
                    private=config.hub_private,
                    token=token
                )
                logger.success("Merged model pushed to HuggingFace Hub successfully")
            
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    load_dotenv('.env')
    
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    training_dir = Path('/home/grahama/dev/vllm_lora/training_output')
    adapter_path = str(Path(training_dir, 'Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter/final_model'))
    
    # Path to your markdown template
    template_path = Path('templates/model_card_template.md')
    
    push_adapter_to_hub(
        adapter_path=adapter_path,
        hub_model_id="grahamaco/Meta-Llama-3.1-8B-Instruct-bnb-4bit_touch-rugby-rules_adapter",
        token=token,
        private=True,
        model_card_template=str(template_path)
    )