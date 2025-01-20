import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from .common import Evaluator
import pdb

class Faithfulness(Evaluator):
    def __init__(self, model, step=3136, substrate_fn=torch.zeros_like, postprocess=None):
        """Faithfulness metric implementation for patch-based evaluation.
        
        Implementation based on definition in:
        "Synthetic Benchmarks for Scientific Research in Explainable Machine Learning"
        (Liu et al., 2021) https://arxiv.org/pdf/2106.12543
        
        Args:
            model (nn.Module): Model being explained
            step (int): Number of pixels in each patch (e.g. 56*56=3136 for 56x56 patches)
            substrate_fn (callable): Function to replace removed patches
            postprocess (callable, optional): Post-processing function for model output
        """
        super(Faithfulness, self).__init__(model, postprocess)
        self.step = step
        self.substrate_fn = substrate_fn

    def get_patch_indices(self, height, width):
        """Generate indices for all possible patches.
        
        Args:
            height (int): Image height
            width (int): Image width
            
        Returns:
            list: List of (start_h, start_w, end_h, end_w) tuples for each patch
        """
        patch_size = int(np.sqrt(self.step))  # e.g., 56 for step=3136
        patches = []
        for h in range(0, height, patch_size):
            for w in range(0, width, patch_size):
                h_end = min(h + patch_size, height)
                w_end = min(w + patch_size, width)
                patches.append((h, w, h_end, w_end))
        return patches

    def get_patch_importance(self, Z, patch_coords):
        """Calculate importance score for a patch based on attribution values.
        
        Args:
            Z (Tensor): Attribution map
            patch_coords (tuple): (start_h, start_w, end_h, end_w)
            
        Returns:    
            float: Importance score for the patch
        """
        h_start, w_start, h_end, w_end = patch_coords
        patch_attrs = Z[:, :, h_start:h_end, w_start:w_end]
        return patch_attrs.mean().item()

    def forward(self, X, Z, kwargs={}, return_dict=False):
        """Calculate faithfulness score using all patches systematically.
        
        Args:
            X (Tensor): Input images (B, C, H, W)
            Z (Tensor): Attribution maps (B, 1, H, W)
            kwargs (dict): Additional arguments for model forward pass
            return_dict (bool): Whether to return detailed results
        """
        self.model.eval()
        batch_size = X.shape[0]
        height, width = X.shape[2], X.shape[3]
        
        # Get original predictions
        with torch.no_grad():
            orig_preds = self.model(X, **kwargs)
            # pdb.set_trace()
            if self.postprocess is not None:
                orig_preds = self.postprocess(orig_preds)
            orig_conf, pred_cls = torch.max(torch.softmax(orig_preds, dim=1), dim=1)

        # Get all patches systematically
        all_patches = self.get_patch_indices(height, width)
        
        # Initialize arrays for correlations
        faithfulness_scores = []
        attribution_scores = []
        impact_scores = []
        
        # For each image in batch
        for b in range(batch_size):
            # Evaluate all patches systematically
            attr_scores = []
            impacts = []
            
            for patch_coords in all_patches:
                # Calculate patch importance from attribution map
                attr_score = self.get_patch_importance(Z[b:b+1], patch_coords)
                attr_scores.append(attr_score)
                
                # Create modified image with patch removed
                X_mod = X[b:b+1].clone()
                h_start, w_start, h_end, w_end = patch_coords
                X_mod[:, :, h_start:h_end, w_start:w_end] = self.substrate_fn(
                    X_mod[:, :, h_start:h_end, w_start:w_end])
                
                # Get new prediction
                with torch.no_grad():
                    mod_pred = self.model(X_mod, **{k: v[b:b+1] if torch.is_tensor(v) else v 
                                                   for k, v in kwargs.items()})
                    if self.postprocess is not None:
                        mod_pred = self.postprocess(mod_pred)
                    mod_conf = torch.softmax(mod_pred, dim=1)[0, pred_cls[b]]
                
                # Impact is difference in confidence
                impact = (orig_conf[b] - mod_conf).cpu().item()
                impacts.append(impact)
                # pdb.set_trace()
            
            # Calculate Pearson correlation
            correlation = stats.pearsonr(attr_scores, impacts)[0]
            if np.isnan(correlation) or not np.isfinite(correlation):
                correlation = 0 #Occurs when impact is constant, handled in same way as abacus ai implementation

            faithfulness_scores.append(correlation)
            
            # Store scores for detailed results
            attribution_scores.append(attr_scores)
            impact_scores.append(impacts)
        
        faithfulness_scores = torch.tensor(faithfulness_scores, device=X.device)
        # pdb.set_trace()
        if return_dict:
            return {
                'faithfulness': faithfulness_scores,
                'attribution_scores': attribution_scores,
                'impact_scores': impact_scores,
                'patches': all_patches
            }
        return faithfulness_scores

    def plot_correlation(self, attribution_scores, impact_scores, save_dir='output/faithfulness'):
        """Plot correlation between attribution and impact scores.
        
        Args:
            attribution_scores (list): List of attribution scores
            impact_scores (list): List of impact scores
            save_dir (str): Directory to save plots
        """
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i, (attr, impact) in enumerate(zip(attribution_scores, impact_scores)):
            plt.figure(figsize=(8, 6))
            plt.scatter(attr, impact, alpha=0.5)
            plt.xlabel('Mean Patch Attribution')
            plt.ylabel('Patch Removal Impact')
            
            # Add correlation coefficient
            corr = stats.pearsonr(attr, impact)[0]
            plt.title(f'Patch Faithfulness Correlation: {corr:.3f}')
            
            # Add trend line
            z = np.polyfit(attr, impact, 1)
            p = np.poly1d(z)
            plt.plot(attr, p(attr), "r--", alpha=0.8)
            
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'faithfulness_correlation_{i}.png'))
            plt.close() 
    

def main():
    """Unit test for Faithfulness evaluator using gradient pattern test."""
    import torch.nn as nn
    
    class GradientScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_size = 224
            self.patch_size = 56
            self.n_patches = self.img_size // self.patch_size
            
            gradient = torch.linspace(0, 1, self.n_patches)
            gradient = gradient.view(1, -1).repeat(self.n_patches, 1)
            self.register_buffer('gradient_pattern', gradient)
            
        def forward(self, x, **kwargs):
            B = x.size(0)
            patches = x.view(B, 1, self.n_patches, self.patch_size, 
                           self.n_patches, self.patch_size)
            patch_means = patches.mean(dim=(3,5)).squeeze(1)
            match_score = (patch_means * self.gradient_pattern).sum(dim=(1,2)) / self.n_patches
            return torch.stack([1 - match_score, match_score], dim=1)
    
    # Create test data
    batch_size = 4
    patch_size = 56
    n_patches = 224 // patch_size  # Should be 4
    noise_level = 0.1  # Standard deviation of noise for attribution
    
    # Create images with clean gradient pattern
    X = torch.zeros(batch_size, 1, 224, 224)
    for b in range(batch_size):
        for i in range(n_patches):
            for j in range(n_patches):
                h_start = i * patch_size
                w_start = j * patch_size
                # Clean gradient value
                patch_value = j / (n_patches - 1)
                X[b, :, h_start:h_start+patch_size, w_start:w_start+patch_size] = patch_value
    
    def random_attribution(x, model, patch_size=56, noise_level=0.1):
        """Random attribution (phi_1) with noise"""
        B = x.size(0)
        H, W = x.shape[2], x.shape[3]
        n_patches = H // patch_size
        
        # Create random patch-level attributions
        patch_attributions = torch.rand(B, 1, n_patches, n_patches)
        
        # Add noise to attributions
        noise = torch.randn_like(patch_attributions) * noise_level
        patch_attributions = patch_attributions + noise
        
        # Normalize to [0,1] range
        patch_attributions = (patch_attributions - patch_attributions.min()) / (
            patch_attributions.max() - patch_attributions.min())
        
        # Expand to full image size
        attr = torch.zeros(B, 1, H, W)
        for i in range(n_patches):
            for j in range(n_patches):
                h_start = i * patch_size
                h_end = h_start + patch_size
                w_start = j * patch_size
                w_end = w_start + patch_size
                attr[:, :, h_start:h_end, w_start:w_end] = patch_attributions[:, :, i, j].view(B, 1, 1, 1)
        
        return attr.to(x.device)
    
    def gradient_attribution(x, model, patch_size=56, noise_level=0.1):
        """Ground truth gradient attribution with noise"""
        B = x.size(0)
        H, W = x.shape[2], x.shape[3]
        n_patches = H // patch_size
        
        # Create gradient pattern attributions
        patch_attributions = torch.zeros(B, 1, n_patches, n_patches)
        for i in range(n_patches):
            for j in range(n_patches):
                patch_attributions[:, :, i, j] = j / (n_patches - 1)
        
        # Add noise to attributions
        noise = torch.randn_like(patch_attributions) * noise_level
        patch_attributions = patch_attributions + noise
        
        # Normalize to [0,1] range
        patch_attributions = (patch_attributions - patch_attributions.min()) / (
            patch_attributions.max() - patch_attributions.min())
        
        # Expand to full image size
        attr = torch.zeros(B, 1, H, W)
        for i in range(n_patches):
            for j in range(n_patches):
                h_start = i * patch_size
                h_end = h_start + patch_size
                w_start = j * patch_size
                w_end = w_start + patch_size
                attr[:, :, h_start:h_end, w_start:w_end] = patch_attributions[:, :, i, j].view(B, 1, 1, 1)
        
        return attr.to(x.device)

    # Initialize model and evaluator
    model = GradientScorer()
    faithfulness = Faithfulness(model, step=56*56)
    
    # Test random attribution
    print("Testing random attribution (phi_1)...")
    Z_random = random_attribution(X, model, patch_size=56, noise_level=noise_level)
    results_random = faithfulness(X, Z_random, return_dict=True)
    print(f"Random attribution faithfulness scores: {results_random['faithfulness']}")
    
    # Test gradient attribution
    print("\nTesting gradient attribution (phi_2)...")
    Z_gradient = gradient_attribution(X, model, patch_size=56, noise_level=noise_level)
    results_gradient = faithfulness(X, Z_gradient, return_dict=True)
    print(f"Gradient attribution faithfulness scores: {results_gradient['faithfulness']}")
    
    # Save visualization of noisy input and attributions
    def save_visualizations(X, Z_random, Z_gradient, save_dir='output/faithfulness_test'):
        import os
        from torchvision.utils import save_image
        os.makedirs(save_dir, exist_ok=True)
        
        # Save first image from batch
        save_image(X[0], os.path.join(save_dir, 'noisy_input.png'))
        # save_image(X_clean[0], os.path.join(save_dir, 'clean_input.png'))
        save_image(Z_random[0], os.path.join(save_dir, 'random_attribution.png'))
        save_image(Z_gradient[0], os.path.join(save_dir, 'gradient_attribution.png'))
    
    save_visualizations(X, Z_random, Z_gradient)
    
    # Verify attribution consistency within patches
    def verify_patch_consistency(Z, patch_size=56):
        batch_size = Z.shape[0]
        H, W = Z.shape[2], Z.shape[3]
        
        for b in range(batch_size):
            for h in range(0, H, patch_size):
                for w in range(0, W, patch_size):
                    h_end = min(h + patch_size, H)
                    w_end = min(w + patch_size, W)
                    patch = Z[b:b+1, :, h:h_end, w:w_end]
                    first_value = patch[0,0,0,0].item()
                    
                    assert torch.allclose(
                        patch, 
                        first_value * torch.ones_like(patch)
                    ), f"Attribution values not consistent within patch for batch {b}!"
    
    print("\nVerifying patch consistency...")
    verify_patch_consistency(Z_random)
    verify_patch_consistency(Z_gradient)
    print("Patch consistency verified!")
    
    # Plot correlations
    print("\nPlotting correlations...")
    save_dir = 'output/faithfulness_test'
    
    faithfulness.plot_correlation(
        results_random['attribution_scores'],
        results_random['impact_scores'],
        save_dir=f'{save_dir}/random'
    )
    
    faithfulness.plot_correlation(
        results_gradient['attribution_scores'],
        results_gradient['impact_scores'],
        save_dir=f'{save_dir}/gradient'
    )
    
    # Compare results
    random_mean = results_random['faithfulness'].mean().item()
    gradient_mean = results_gradient['faithfulness'].mean().item()
    
    print(f"\nMean faithfulness scores:")
    print(f"Random attribution: {random_mean:.3f}")
    print(f"Gradient attribution: {gradient_mean:.3f}")
    
    assert gradient_mean > random_mean, "Gradient attribution should have higher faithfulness!"
    print("\nAll tests passed! Gradient attribution shows higher faithfulness as expected.")

if __name__ == '__main__':
    main()
    

    