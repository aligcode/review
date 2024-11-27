import torch
import torch.nn as nn

class ImageCompletionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, num_layers, num_patches):
        super(ImageCompletionTransformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Linear projection for patch embeddings
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, embed_dim)

        # Learnable positional encodings
        self.positional_embedding = nn.Parameter(torch.randn(num_patches, embed_dim))

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Transformer decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size * patch_size * 3)
        )

    def forward(self, patches, mask):
        """
        Args:
            patches (torch.Tensor): Patch embeddings of shape (batch_size, num_patches, embed_dim)
            mask (torch.Tensor): Binary mask of shape (batch_size, num_patches) (1 = visible, 0 = masked)
        """
        batch_size, num_patches, _ = patches.size()

        # Replace masked patches with mask token
        mask_token = self.mask_token.expand(batch_size, num_patches, -1)  # Shape: (batch_size, num_patches, embed_dim)
        patches = torch.where(mask.unsqueeze(-1), patches, mask_token)  # Replace masked locations

        # Add positional encodings
        patches = patches + self.positional_embedding.unsqueeze(0)  # Broadcast positional embeddings

        # Encode patches
        encoded_patches = self.encoder(patches)  # Shape: (batch_size, num_patches, embed_dim)

        # Decode patches
        reconstructed_patches = self.decoder(encoded_patches)  # Shape: (batch_size, num_patches, patch_dim)

        return reconstructed_patches
    

def image_to_patches(image, patch_size):
    """
    Converts an image into non-overlapping patches.
    Args:
        image (torch.Tensor): Input image of shape (batch_size, channels, height, width)
        patch_size (int): Size of each patch (P x P)
    Returns:
        patches (torch.Tensor): Flattened patches of shape (batch_size, num_patches, patch_dim)
    """
    batch_size, channels, height, width = image.size()
    assert height % patch_size == 0 and width % patch_size == 0, "Image dimensions must be divisible by patch size"
    
    # Compute the number of patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = channels * patch_size * patch_size

    # Extract patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, num_patches, patch_size, patch_size)
    
    # Flatten patches
    patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size, num_patches, patch_dim)
    return patches


def patches_to_image(patches, image_size, patch_size):
    """
    Reconstructs an image from non-overlapping patches.
    Args:
        patches (torch.Tensor): Flattened patches of shape (batch_size, num_patches, patch_dim)
        image_size (tuple): Original image size (height, width)
        patch_size (int): Size of each patch (P x P)
    Returns:
        image (torch.Tensor): Reconstructed image of shape (batch_size, channels, height, width)
    """
    batch_size, num_patches, patch_dim = patches.size()
    height, width = image_size
    channels = patch_dim // (patch_size * patch_size)
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size

    # Unflatten patches
    patches = patches.view(batch_size, num_patches, channels, patch_size, patch_size)
    
    # Rearrange patches into the image grid
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    image = patches.view(batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
    image = image.permute(0, 1, 2, 4, 3, 5).contiguous()
    image = image.view(batch_size, channels, height, width)
    return image

# Example usage
reconstructed_image = patches_to_image(patches, (128, 128), patch_size)
print("Reconstructed image shape:", reconstructed_image.shape)  # (batch_size, channels, height, width)

# Example usage
batch_size, channels, height, width = 4, 3, 128, 128
image = torch.randn(batch_size, channels, height, width)
patch_size = 16
patches = image_to_patches(image, patch_size)
print("Patches shape:", patches.shape)  # (batch_size, num_patches, patch_dim)
