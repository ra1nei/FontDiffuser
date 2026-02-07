import torch
import lpips  # pip install lpips

class LPIPSMetric(torch.nn.Module):
    def __init__(self, net='vgg'):
        """
        LPIPS metric (Learned Perceptual Image Patch Similarity)
        using pretrained weights from the official 'lpips' library.

        Args:
            net (str): Backbone network ('alex' | 'vgg' | 'squeeze'). Default: 'vgg'.
        """
        super(LPIPSMetric, self).__init__()
        self.lpips_fn = lpips.LPIPS(net=net)  # True = pretrained weights

    def forward(self, x, y):
        """
        Compute LPIPS distance between two images x and y.
        Inputs should be in [-1, 1] range and shape [B, 3, H, W].
        Returns scalar LPIPS distance (lower = more similar).
        """
        with torch.no_grad():
            dist = self.lpips_fn(x, y)
        return dist.mean()
