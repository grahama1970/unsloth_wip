"""Upload trained models to Hugging Face Hub."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from huggingface_hub import HfApi, create_repo, upload_folder, ModelCard
from loguru import logger
import yaml


class HubUploader:
    """Upload LoRA adapters to Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the uploader."""
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not found in environment")
            
        self.api = HfApi(token=self.token)
        
    async def upload_adapter(
        self,
        adapter_path: Path,
        model_id: str,
        base_model: str,
        training_stats: Optional[Dict[str, Any]] = None,
        private: bool = True,
        create_model_card: bool = True,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Upload a LoRA adapter to Hugging Face Hub.
        
        Args:
            adapter_path: Path to adapter files
            model_id: Repository ID (username/model-name)
            base_model: Base model name
            training_stats: Training statistics
            private: Whether to make repo private
            create_model_card: Whether to create a model card
            tags: Additional tags for the model
            
        Returns:
            Upload results dictionary
        """
        logger.info(f"Uploading adapter to: {model_id}")
        
        try:
            # Create repository
            logger.info("Creating repository...")
            repo_url = create_repo(
                repo_id=model_id,
                token=self.token,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            
            # Prepare files
            upload_path = self._prepare_upload_files(
                adapter_path, 
                base_model,
                training_stats,
                create_model_card,
                tags
            )
            
            # Upload files
            logger.info("Uploading files...")
            upload_info = upload_folder(
                folder_path=str(upload_path),
                repo_id=model_id,
                token=self.token,
                commit_message=f"Upload LoRA adapter trained on {base_model}"
            )
            
            # Cleanup temp files
            if upload_path != adapter_path:
                shutil.rmtree(upload_path)
                
            result = {
                "status": "success",
                "model_id": model_id,
                "url": f"https://huggingface.co/{model_id}",
                "repo_url": repo_url,
                "commit_info": upload_info,
                "private": private
            }
            
            logger.info(f"Upload successful: {result['url']}")
            return result
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_id": model_id
            }
            
    def _prepare_upload_files(
        self,
        adapter_path: Path,
        base_model: str,
        training_stats: Optional[Dict[str, Any]],
        create_model_card: bool,
        tags: Optional[List[str]]
    ) -> Path:
        """Prepare files for upload."""
        # If we need to add files, create a temp directory
        if create_model_card or training_stats:
            temp_path = Path("/tmp") / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(adapter_path, temp_path)
            upload_path = temp_path
        else:
            upload_path = adapter_path
            
        # Create model card if requested
        if create_model_card:
            model_card = self._create_model_card(
                base_model=base_model,
                training_stats=training_stats,
                tags=tags
            )
            with open(upload_path / "README.md", "w") as f:
                f.write(str(model_card))
                
        # Save training stats if provided
        if training_stats:
            with open(upload_path / "training_stats.json", "w") as f:
                json.dump(training_stats, f, indent=2)
                
        # Update adapter config with base model info
        config_path = upload_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            config["base_model_name_or_path"] = base_model
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
        return upload_path
        
    def _create_model_card(
        self,
        base_model: str,
        training_stats: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelCard:
        """Create a model card for the adapter."""
        
        # Default tags
        default_tags = [
            "lora",
            "adapter",
            "unsloth",
            base_model.split("/")[-1],
            "text-generation"
        ]
        
        if tags:
            default_tags.extend(tags)
            
        # Extract key stats
        if training_stats:
            enhancement_stats = training_stats.get("steps", {}).get("enhancement", {})
            training_info = training_stats.get("steps", {}).get("training", {})
        else:
            enhancement_stats = {}
            training_info = {}
            
        # Create card content
        card_content = f"""---
tags:
{yaml.dump(default_tags, default_flow_style=False)}
base_model: {base_model}
language:
- en
library_name: peft
license: apache-2.0
---

# LoRA Adapter for {base_model}

This is a LoRA adapter trained on top of **{base_model}** using the Unsloth library.

## Training Details

This adapter was trained using student-teacher thinking enhancement:
- **Student Model**: {base_model} (same as base)
- **Teacher Model**: Claude (anthropic/max)
- **Enhancement Stats**:
  - Examples enhanced: {enhancement_stats.get('enhanced_examples', 'N/A')}
  - Average iterations: {enhancement_stats.get('average_iterations', 'N/A')}
  - Convergence rate: {enhancement_stats.get('convergence_rate', 0)*100:.1f}%

## Model Details

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Unsloth (optimized for efficiency)

## Usage

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    adapter_name="{os.getenv('HF_USERNAME', 'your-username')}/your-adapter-name",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# For inference
FastLanguageModel.for_inference(model)

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration

"""
        
        if training_info:
            metrics = training_info.get("metrics", {})
            card_content += f"""
- Final Loss: {metrics.get('loss', 'N/A')}
- Training Location: {training_info.get('training_location', 'N/A')}
"""
            
        card_content += """
## Limitations

This is a LoRA adapter and requires the base model to function. The adapter modifies the behavior of the base model but inherits its fundamental capabilities and limitations.

## Citation

If you use this model, please cite:

```bibtex
@misc{unsloth2024,
  title={Unsloth: Efficient LLM Fine-tuning},
  author={Unsloth Team},
  year={2024},
  url={https://github.com/unslothai/unsloth}
}
```
"""
        
        return ModelCard(card_content)
        
    async def create_collection(
        self,
        collection_name: str,
        model_ids: List[str],
        description: str,
        private: bool = False
    ) -> Dict[str, Any]:
        """Create a collection of related models."""
        logger.info(f"Creating collection: {collection_name}")
        
        try:
            # Create collection using the API
            collection = self.api.create_collection(
                title=collection_name,
                description=description,
                namespace=os.getenv("HF_USERNAME"),
                private=private
            )
            
            # Add models to collection
            for model_id in model_ids:
                self.api.add_collection_item(
                    collection_slug=collection.slug,
                    item_id=model_id,
                    item_type="model"
                )
                
            return {
                "status": "success",
                "collection_name": collection_name,
                "collection_url": f"https://huggingface.co/collections/{collection.slug}",
                "models_added": len(model_ids)
            }
            
        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


async def main():
    """Example usage."""
    uploader = HubUploader()
    
    # Upload adapter
    result = await uploader.upload_adapter(
        adapter_path=Path("./outputs/adapter"),
        model_id="username/my-cool-adapter",
        base_model="unsloth/Phi-3.5-mini-instruct",
        training_stats={
            "steps": {
                "enhancement": {
                    "enhanced_examples": 1000,
                    "average_iterations": 2.3,
                    "convergence_rate": 0.87
                }
            }
        },
        tags=["student-teacher", "enhanced-thinking"]
    )
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())