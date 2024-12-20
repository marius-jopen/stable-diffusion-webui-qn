"""
Configuration settings for quantum noise generation and processing
"""

# Default RNG mode
DEFAULT_RNG_MODE = "classic"

# Quantum noise file path (relative to repo root)
QUANTUM_NOISE_PATH = r"input_quantum-noise\qn-full-high-latent.pt"

# Blend settings for quantum and standard noise
BLEND_SETTINGS = {
    # First step blending (randnCustom)
    "first_step": {
        "blend_ratio": 0.0,     # Range: 0.0 to 1.0 (0.0 = pure quantum, 1.0 = pure standard)
        "blend_mode": "normal"   # Options: "normal", "screen", "multiply", "difference"
    },
    
    # Subsequent steps blending (randn_without_seedCustom)
    "subsequent_steps": {
        "blend_ratio": 0.0,     # Range: 0.0 to 1.0 (0.0 = pure quantum, 1.0 = pure standard)
        "blend_mode": "normal"   # Options: "normal", "screen", "multiply", "difference"
    }
}

# Noise processing parameters
NOISE_SETTINGS = {
    # Basic scaling
    "scale_y": 1.0,
    
    # Normalization settings
    "normalization": "gaussian",  # Options: "none", "gaussian", "minmax", "standard"
    "norm_strength": 0.0,        # Range: 0.0 to 1.0
    
    # Distribution adjustments
    "power": 1.0,               # Range: typically 0.1 to 2.0
    "gaussian_mix": 0.0,        # Range: 0.0 to 1.0
    
    # Frequency domain filters
    "high_pass": 0.0,           # Range: 0.0 to 1.0
    "low_pass": 1.0,            # Range: 0.0 to 1.0
    
    # Multi-scale settings
    "num_scales": 1             # Range: 1 to 5 (integer)
}
