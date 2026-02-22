# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CriticModel: LLM + Cumulative Average Pooling + Linear Classifier

This model is designed to be trained separately and then loaded as a checkpoint
for use in G2RPOCriticsTrainer for policy training.
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class CriticModel(nn.Module):
    """
    Critic model that uses LLM embeddings + cumulative average pooling + linear classifier.
    
    For a sequence, the LLM outputs hidden states of shape (B, L, hidden_size).
    We apply cumulative average pooling: for position j, we average embeddings from 0 to j.
    Then we pass through a linear classifier with softmax to get token-wise prediction distributions.
    
    This model can be:
    1. Trained separately using train_critics.py
    2. Saved with model.save_pretrained() or torch.save(model.state_dict(), ...)
    3. Loaded in G2RPOCriticsTrainer for policy training
    
    Example usage for training:
    ```python
    from trl import CriticModel
    
    model = CriticModel("Qwen/Qwen2-0.5B-Instruct")
    # ... train the model ...
    model.save_pretrained("path/to/checkpoint")
    ```
    
    Example usage for loading:
    ```python
    from trl import CriticModel
    
    model = CriticModel.from_pretrained("path/to/checkpoint")
    ```
    """
    
    def __init__(self, llm_path: str, vocab_size: int = None, **llm_kwargs):
        """
        Initialize the CriticModel.
        
        Args:
            llm_path: Path to the LLM model (HuggingFace model ID or local path)
            vocab_size: Output vocabulary size for the classifier. If None, uses LLM's vocab_size.
            **llm_kwargs: Additional keyword arguments passed to AutoModelForCausalLM.from_pretrained()
        """
        super().__init__()
        # Load LLM backbone
        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, **llm_kwargs)
        
        # Store the llm_path for save_pretrained
        self.llm_path = llm_path
        
        # Get hidden size from LLM config
        self.hidden_size = self.llm.config.hidden_size
        
        # Use vocab_size from LLM if not provided
        if vocab_size is None:
            vocab_size = self.llm.config.vocab_size
        self.vocab_size = vocab_size
        
        # Linear classifier for token-wise prediction
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)
    
    @property
    def config(self):
        """Expose the underlying LLM's config for DeepSpeed compatibility."""
        return self.llm.config
        
    def cumulative_avg_pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply cumulative average pooling over sequence dimension.
        
        For position j, compute mean of hidden states from position 0 to j (inclusive).
        
        Args:
            hidden_states: (B, L, hidden_size)
        
        Returns:
            pooled: (B, L, hidden_size) where pooled[i,j,k] = mean(hidden_states[i,0:j+1,k])
        """
        B, L, H = hidden_states.shape
        # Compute cumulative sum along sequence dimension
        cumsum = torch.cumsum(hidden_states, dim=1)  # (B, L, H)
        # Create position indices for division (1, 2, 3, ..., L)
        positions = torch.arange(1, L + 1, device=hidden_states.device, dtype=hidden_states.dtype)
        positions = positions.view(1, L, 1)  # (1, L, 1) for broadcasting
        # Divide cumsum by position to get cumulative average
        pooled = cumsum / positions  # (B, L, H)
        return pooled
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """
        Forward pass through critic model.
        
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
        
        Returns:
            CausalLMOutputWithPast with logits of shape (B, L, vocab_size)
        """
        # Get hidden states from LLM
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]  # (B, L, hidden_size)
        
        # Apply cumulative average pooling
        pooled = self.cumulative_avg_pool(hidden_states)  # (B, L, hidden_size)
        
        # Pass through linear classifier (softmax is applied during entropy computation)
        logits = self.classifier(pooled)  # (B, L, vocab_size)
        
        # Return in same format as CausalLM output
        return CausalLMOutputWithPast(logits=logits, hidden_states=outputs.hidden_states)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the critic model to a directory.
        
        This saves:
        - The LLM backbone (in a subdirectory 'llm/')
        - The classifier weights (classifier.pt)
        - A config file (critic_config.json)
        
        Args:
            save_directory: Directory to save the model to
            **kwargs: Additional arguments passed to llm.save_pretrained()
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save LLM backbone
        llm_save_path = os.path.join(save_directory, "llm")
        self.llm.save_pretrained(llm_save_path, **kwargs)
        
        # Save classifier weights
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        
        # Save critic config
        config = {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "llm_path": self.llm_path,
        }
        config_path = os.path.join(save_directory, "critic_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **llm_kwargs):
        """
        Load a critic model from a directory.
        
        Args:
            load_directory: Directory containing the saved model
            **llm_kwargs: Additional arguments passed to the LLM loading
            
        Returns:
            CriticModel instance with loaded weights
        """
        import os
        import json
        
        # Load critic config
        config_path = os.path.join(load_directory, "critic_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load LLM from the saved subdirectory
        llm_path = os.path.join(load_directory, "llm")
        
        # Create model instance
        model = cls(llm_path, vocab_size=config["vocab_size"], **llm_kwargs)
        
        # Load classifier weights
        classifier_path = os.path.join(load_directory, "classifier.pt")
        classifier_state = torch.load(classifier_path, map_location="cpu")
        model.classifier.load_state_dict(classifier_state)
        
        return model

