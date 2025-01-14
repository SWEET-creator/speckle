import torch
import torch.nn as nn
from transformers import ViTModel

class ViTBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTBinaryClassifier, self).__init__()
        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Freeze the pretrained layers (optional)
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Define the MLP head
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Pass input through ViT
        outputs = self.vit(pixel_values=x)
        
        # Get the [CLS] token embedding
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]
        
        # Pass through classifier head
        logits = self.classifier(cls_token_embedding)
        return logits
