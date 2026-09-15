import torch
import torch.nn as nn


class PassthroughEncoder(nn.Module):
    """Stands in for the frozen DINO encoder when the dataset already serves its latents.

    `VWorldModel` only needs `name`, `emb_dim`, `latent_ndim`, `patch_size` and a forward
    that maps the dataset's "visual" tensor to (b*t, patches, emb_dim). The name must not
    contain "dino", so the model's encoder_transform is the identity rather than a Resize.
    One dummy parameter keeps the trainer's (unused) encoder optimizer constructible.
    """

    def __init__(self, emb_dim=384, patch_size=14, name="cached_latents"):
        super().__init__()
        self.name, self.emb_dim, self.patch_size, self.latent_ndim = name, emb_dim, patch_size, 2
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return x.float()
