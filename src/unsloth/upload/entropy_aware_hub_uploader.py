"""Entropy-aware HuggingFace Hub uploader with metadata.
Module: entropy_aware_hub_uploader.py
Description: Upload models with entropy training metadata and visualizations

External Dependencies:
- huggingface_hub: https://huggingface.co/docs/huggingface_hub/
- torch: https://pytorch.org/docs/stable/

Sample Input:
>>> uploader = EntropyAwareHubUploader()
>>> result = await uploader.upload_with_metadata(
...     adapter_path="./adapter",
...     hub_id="username/model-entropy",
...     metadata={"entropy_scale": 2.0}
... )

Expected Output:
>>> result['url']
'https://huggingface.co/username/model-entropy'
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from loguru import logger

from unsloth.upload.hub_uploader import HubUploader


class EntropyAwareHubUploader(HubUploader):
    """Extended hub uploader with entropy-specific features."""
    
    def __init__(self):
        """Initialize entropy-aware uploader."""
        super().__init__()
        self.api = HfApi()
    
    async def upload_with_metadata(
        self,
        adapter_path: str,
        hub_id: str,
        metadata: Dict,
        include_visualizations: bool = True,
        visualization_dir: Optional[str] = None,
        private: bool = False,
        create_pr: bool = False
    ) -> Dict:
        """Upload model with entropy training metadata.
        
        Args:
            adapter_path: Path to adapter/model
            hub_id: HuggingFace Hub ID
            metadata: Entropy training metadata
            include_visualizations: Include visualization files
            visualization_dir: Directory with visualizations
            private: Make repo private
            create_pr: Create PR instead of direct push
            
        Returns:
            Upload result with URL
        """
        adapter_path = Path(adapter_path)
        
        # Prepare upload directory
        upload_dir = Path("/tmp/entropy_upload") / hub_id.replace("/", "_")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy adapter files
        logger.info("Copying adapter files...")
        for file in adapter_path.glob("*"):
            if file.is_file():
                shutil.copy2(file, upload_dir / file.name)
        
        # Add entropy metadata to config
        config_path = upload_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
        
        config["entropy_training"] = metadata
        config["upload_timestamp"] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create comprehensive model card
        model_card = self._create_entropy_model_card(hub_id, metadata)
        with open(upload_dir / "README.md", 'w') as f:
            f.write(model_card)
        
        # Copy visualizations if requested
        if include_visualizations and visualization_dir:
            vis_dir = Path(visualization_dir)
            if vis_dir.exists():
                logger.info("Including visualizations...")
                vis_upload_dir = upload_dir / "visualizations"
                vis_upload_dir.mkdir(exist_ok=True)
                
                for vis_file in vis_dir.glob("*.html"):
                    shutil.copy2(vis_file, vis_upload_dir / vis_file.name)
        
        # Create repository
        logger.info(f"Creating repository: {hub_id}")
        try:
            create_repo(
                repo_id=hub_id,
                private=private,
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Repo creation warning: {e}")
        
        # Upload files
        logger.info("Uploading files to Hub...")
        try:
            url = upload_folder(
                folder_path=str(upload_dir),
                repo_id=hub_id,
                create_pr=create_pr,
                commit_message="Upload entropy-aware trained model"
            )
            
            result = {
                "status": "success",
                "url": f"https://huggingface.co/{hub_id}",
                "upload_path": str(upload_dir),
                "files_uploaded": len(list(upload_dir.rglob("*")))
            }
            
            logger.info(f" Model uploaded successfully: {result['url']}")
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            result = {
                "status": "failed",
                "error": str(e)
            }
        
        # Cleanup
        shutil.rmtree(upload_dir, ignore_errors=True)
        
        return result
    
    def _create_entropy_model_card(self, hub_id: str, metadata: Dict) -> str:
        """Create comprehensive model card with entropy information."""
        model_name = hub_id.split("/")[-1]
        
        card = f"""---
license: apache-2.0
tags:
- generated_from_trainer
- entropy-aware
- unsloth
- lora
library_name: transformers
base_model: {metadata.get('base_model', 'unsloth/Phi-3.5-mini-instruct')}
---

# {model_name}

This model was trained using **entropy-aware training** with the Unsloth framework.

## Training Details

### Entropy Configuration
- **Weight Scale**: {metadata.get('entropy_config', {}).get('weight_scale', 2.0)}
- **Entropy Threshold**: {metadata.get('entropy_config', {}).get('threshold', 0.5)}
- **Dynamic Weighting**: {metadata.get('entropy_config', {}).get('dynamic_weighting', True)}

### Training Type
- **Method**: {metadata.get('training_type', 'entropy_aware')}
- **Pipeline Version**: {metadata.get('pipeline_version', '1.0.0')}
- **Timestamp**: {metadata.get('timestamp', 'N/A')}

## What is Entropy-Aware Training?

Entropy-aware training focuses computational resources on high-uncertainty tokens during training. 
This approach:
- Improves learning on difficult/ambiguous parts of the data
- Reduces overfitting on low-entropy (easy) tokens
- Results in better generalization

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{metadata.get('base_model', 'unsloth/Phi-3.5-mini-instruct')}",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "{hub_id}")

# Use for inference
tokenizer = AutoTokenizer.from_pretrained("{hub_id}")
```

## Performance

The model was trained with entropy-aware loss weighting, which typically results in:
- Better handling of ambiguous or complex queries
- Improved reasoning on high-uncertainty tasks
- More calibrated confidence in responses

## Visualizations

Check the `visualizations/` folder for:
- Token-level entropy heatmaps
- Training entropy progression
- Entropy distribution analysis

## Citation

If you use this model, please cite:

```bibtex
@misc{{unsloth2024entropy,
  title={{Entropy-Aware Training with Unsloth}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/{hub_id}}}
}}
```

## Framework

Trained using [Unsloth](https://github.com/unslothai/unsloth) with entropy-aware enhancements.
"""
        
        return card


if __name__ == "__main__":
    # Validation
    async def validate():
        uploader = EntropyAwareHubUploader()
        
        # Test metadata creation
        metadata = {
            "training_type": "entropy_aware",
            "entropy_config": {
                "weight_scale": 2.0,
                "threshold": 0.5
            }
        }
        
        # Test model card generation
        card = uploader._create_entropy_model_card("test/model", metadata)
        assert "entropy-aware training" in card
        assert "Weight Scale: 2.0" in card
        
        print(" Entropy-aware hub uploader validation passed")
    
    asyncio.run(validate())