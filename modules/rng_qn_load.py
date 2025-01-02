import torch
import os
from pathlib import Path
from modules.rng_qn_config import QUANTUM_NOISE_PATH, NOISE_SETTINGS

# Primary quantum noise directory - relative to repo root
_NOISE_PATH = QUANTUM_NOISE_PATH
NOISE_FILE = os.path.join(*_NOISE_PATH.split("\\"))

# Add at the top with other globals
global _current_norm_strength
_current_norm_strength = NOISE_SETTINGS["norm_strength"]

"""
Input: noise tensor, normalization method (str), strength (float)
Output: normalized noise tensor
Used by: prepare_quantum_noise()
Purpose: Normalizes quantum noise using different statistical methods
"""
def normalize_noise(noise, method="none", strength=1.0):
    """Different ways to normalize the noise distribution with controllable strength"""
    if method == "none":
        return noise
        
    elif method == "gaussian":
        # Makes it behave more like standard normal distribution
        normalized = (noise - noise.mean()) / noise.std()
        return (1 - strength) * noise + strength * normalized
        
    elif method == "minmax":
        # Scales values between 0 and 1
        normalized = (noise - noise.min()) / (noise.max() - noise.min())
        return (1 - strength) * noise + strength * normalized
        
    elif method == "standard":
        # Centers around mean and scales by range
        normalized = (noise - noise.mean()) / (noise.max() - noise.min())
        return (1 - strength) * noise + strength * normalized
    
    return noise

"""
Input: noise tensor, power (float)
Output: transformed noise tensor
Used by: prepare_quantum_noise()
Purpose: Adjusts the intensity distribution of the noise by applying power transformation
"""
def power_transform(noise, power=1.0):
    """Changes the "sharpness" of the noise"""
    return torch.sign(noise) * torch.abs(noise) ** power

"""
Input: noise tensor, mix_ratio (float)
Output: blended noise tensor
Used by: prepare_quantum_noise()
Purpose: Combines quantum noise with traditional Gaussian noise at specified ratio
"""
def mix_with_gaussian(noise, mix_ratio=0.5):
    """Blends quantum noise with traditional Gaussian noise"""
    gaussian = torch.randn_like(noise)
    return (1 - mix_ratio) * noise + mix_ratio * gaussian

"""
Input: noise tensor, high_pass/low_pass cutoffs (float)
Output: frequency-filtered noise tensor
Used by: prepare_quantum_noise()
Purpose: Filters noise in frequency domain to modify its spatial characteristics
"""
def frequency_modify(noise, high_pass=0.0, low_pass=1.0):
    """Modifies noise in frequency domain"""
    # Convert to frequency domain
    fft = torch.fft.fft2(noise)
    
    # Create frequency mask
    h, w = noise.shape[-2:]
    Y, X = torch.meshgrid(torch.fft.fftfreq(h), torch.fft.fftfreq(w))
    frequencies = torch.sqrt(X**2 + Y**2)
    
    # Apply frequency filters
    mask = (frequencies >= high_pass) & (frequencies <= low_pass)
    fft_filtered = fft * mask
    
    # Convert back to spatial domain
    return torch.fft.ifft2(fft_filtered).real

"""
Input: noise tensor, number of scales (int)
Output: multi-scale noise tensor
Used by: prepare_quantum_noise()
Purpose: Creates a composite noise by combining different spatial scales
"""
def create_multiscale_noise(noise, num_scales=3):
    """Creates and mixes noise at different scales"""
    scale_weights = [1.0/num_scales] * num_scales
    scales = []
    current_noise = noise
    
    for i in range(num_scales):
        # Downsample and upsample to get different scale
        downscaled = torch.nn.functional.avg_pool2d(current_noise, 2**i)
        upscaled = torch.nn.functional.interpolate(
            downscaled, 
            size=noise.shape[-2:],
            mode='bilinear'
        )
        scales.append(upscaled * scale_weights[i])
    
    return sum(scales)

"""
Input: none
Output: raw quantum noise tensor or None if loading fails
Used by: load_quantum_noise()
Purpose: Loads raw quantum noise data from the specified file path
"""
def load_raw_quantum_noise(selected_file=None):
    """Load the raw quantum noise from file"""
    # Use the selected file if provided, otherwise fall back to default NOISE_FILE
    noise_path = os.path.join("input_quantum-noise", selected_file) if selected_file else NOISE_FILE
    print(f"[QUANTUM NOISE] Loading noise file: {selected_file if selected_file else 'default'}")
    print(f"[QUANTUM NOISE] Full path: {noise_path}")
    
    if not os.path.exists(noise_path):
        print(f"[QUANTUM NOISE] Error: File not found at: {noise_path}")
        return None
        
    try:
        saved_noise = torch.load(noise_path, map_location='cpu')
        print(f"[QUANTUM NOISE] Successfully loaded noise from file: {os.path.basename(noise_path)}")
        print(f"[QUANTUM NOISE] Noise shape: {saved_noise.shape}")
        return saved_noise
    except Exception as e:
        print(f"[QUANTUM NOISE] Error loading noise file {os.path.basename(noise_path)}: {str(e)}")
        return None

"""
Input: saved_noise tensor, target shape, device, and various modification parameters
Output: processed quantum noise tensor
Used by: load_quantum_noise() and external calls
Purpose: Main processing pipeline for quantum noise, applying all modifications
"""
def prepare_quantum_noise(saved_noise, shape, device,
                        scale_y=NOISE_SETTINGS["scale_y"],
                        normalization=NOISE_SETTINGS["normalization"],
                        norm_strength=None,
                        power=NOISE_SETTINGS["power"],
                        gaussian_mix=NOISE_SETTINGS["gaussian_mix"],
                        high_pass=NOISE_SETTINGS["high_pass"],
                        low_pass=NOISE_SETTINGS["low_pass"],
                        num_scales=NOISE_SETTINGS["num_scales"]):
    """Enhanced quantum noise preparation with all modifications"""
    
    global _current_norm_strength
    
    if norm_strength is None:
        norm_strength = _current_norm_strength
    
    if saved_noise is None:
        print("[QUANTUM NOISE] No valid noise data provided to prepare_quantum_noise")
        return torch.randn(shape, device=device)
    
    # Handle batch dimension properly
    if len(saved_noise.shape) == 5:  # [B, 1, C, H, W]
        saved_noise = saved_noise.squeeze(1)  # Remove single channel dim -> [B, C, H, W]
    
    # If we have a batch dimension but don't need it, take first batch
    if len(saved_noise.shape) == 4 and len(shape) == 3:  # [B, C, H, W] -> [C, H, W]
        saved_noise = saved_noise[0]
    
    # Resize if dimensions don't match
    if saved_noise.shape[-3:] != shape[-3:]:  # Compare [C, H, W]
        if len(saved_noise.shape) == 4:  # [B, C, H, W]
            B = saved_noise.shape[0]
            saved_noise = torch.nn.functional.interpolate(
                saved_noise,
                size=shape[-2:],  # Only pass H,W
                mode='bilinear'
            )
        else:  # [C, H, W]
            saved_noise = torch.nn.functional.interpolate(
                saved_noise.unsqueeze(0),  # Add batch dim
                size=shape[-2:],  # Only pass H,W
                mode='bilinear'
            ).squeeze(0)  # Remove batch dim
    
    noise = saved_noise * scale_y
    
    # Apply all modifications as before
    noise = normalize_noise(noise, method=normalization, strength=norm_strength)
    print(f"[QUANTUM NOISE] After normalization - mean: {noise.mean():.4f}, std: {noise.std():.4f}")
    
    if power != 1.0:
        noise = power_transform(noise, power)
    
    if high_pass > 0 or low_pass < 1.0:
        noise = frequency_modify(noise, high_pass, low_pass)
    
    if num_scales > 1:
        noise = create_multiscale_noise(noise, num_scales)
    
    if gaussian_mix > 0:
        noise = mix_with_gaussian(noise, gaussian_mix)
    
    # Ensure final shape matches target shape
    if len(shape) == 4 and len(noise.shape) == 3:  # Need batch dim
        noise = noise.unsqueeze(0)
        if shape[0] > 1:  # Need multiple batches
            noise = noise.expand(shape[0], -1, -1, -1)
    
    return noise.to(device)

"""
Input: target shape and device
Output: prepared quantum noise tensor
Used by: External calls (main entry point)
Purpose: Convenience function that combines loading and preparation steps
"""
def load_quantum_noise(shape, device, selected_file=None):
    """Load and prepare quantum noise from file"""
    raw_noise = load_raw_quantum_noise(selected_file)
    return prepare_quantum_noise(raw_noise, shape, device)



def blend_noise(quantum_noise, standard_noise, blend_ratio, mode="difference"):
    """Blends quantum and standard noise using different blend modes
    Args:
        quantum_noise: Quantum noise tensor
        standard_noise: Standard noise tensor
        blend_ratio: 0.0 = pure quantum, 1.0 = pure random
        mode: Blend mode ("normal", "screen", "multiply", "difference")
    """
    if blend_ratio == 0.0:
        return quantum_noise
    if blend_ratio == 1.0:
        return standard_noise

    if mode == "normal":
        return (1 - blend_ratio) * quantum_noise + blend_ratio * standard_noise
    elif mode == "screen":
        # Screen blend: 1 - (1-a)(1-b)
        return 1 - (1 - quantum_noise) * (1 - standard_noise) * blend_ratio
    elif mode == "multiply":
        # Interpolate between quantum and (quantum * standard)
        return quantum_noise * (1 - blend_ratio + blend_ratio * standard_noise)
    elif mode == "difference":
        # Interpolate between quantum and |quantum - standard|
        return quantum_noise * (1 - blend_ratio) + torch.abs(quantum_noise - standard_noise) * blend_ratio
    else:
        print(f"[QUANTUM NOISE] Unknown blend mode {mode}, falling back to normal")
        return (1 - blend_ratio) * quantum_noise + blend_ratio * standard_noise
