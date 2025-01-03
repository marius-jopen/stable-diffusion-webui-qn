from modules import devices, rng_philox, shared
from modules.rng_qn_load import load_raw_quantum_noise, prepare_quantum_noise, load_quantum_noise, blend_noise
from modules.rng_qn_config import DEFAULT_RNG_MODE, BLEND_SETTINGS

import torch


"""
RANDOM FUNCTIONS USED BY DEFAULT BY STABLE DIFFUSION
"""

"""
Input: seed (int), shape (tuple), generator (optional)
Output: initial noise tensor
Used by: ImageRNG.first()
Purpose: Generates initial noise for the first step of image generation
"""
def randn(seed, shape, generator=None):
    """USED IN FIRST STEP: Generates initial noise with a specific seed"""

    print(f"[NOISE -> randn] randn - seed={seed}, shape={shape}")

    manual_seed(seed)

    if shared.opts.randn_source == "NV":
        print("[NOISE] Using NVIDIA Philox RNG generator")
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        print("[NOISE] Using PyTorch CPU generator")
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    print(f"[NOISE] Using PyTorch generator on {devices.device}")
    return torch.randn(shape, device=devices.device, generator=generator)


"""
Input: shape (tuple), generator (optional)
Output: noise tensor
Used by: ImageRNG.next()
Purpose: Generates noise for subsequent steps using existing generator
"""
def randn_without_seed(shape, generator=None):
    """USED IN STEPS 2-20: Generates new noise using existing generator"""

    print(f"[NOISE -> randn_without_seed] Generating new noise using existing generator and the same seed from first step - shape={shape}")

    if shared.opts.randn_source == "NV":
        print("[NOISE] Using NVIDIA Philox RNG generator")
        return torch.asarray((generator or nv_rng).randn(shape), device=devices.device)

    if shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
        print("[NOISE] Using PyTorch CPU generator")
        return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)

    print(f"[NOISE] Using PyTorch generator on {devices.device}")
    return torch.randn(shape, device=devices.device, generator=generator)


"""
RANDOM FUNCTIONS USED BY DEFAULT BY STABLE DIFFUSION
BUT ACTUALLY NEVER CALLED
"""


"""
Input: seed (int), shape (tuple)
Output: noise tensor
Used by: Not actively used (alternative implementation)
Purpose: Alternative noise generator that doesn't affect global RNG state
"""
def randn_local(seed, shape):
    """Alternative noise generator that uses quantum noise when in custom mode"""
    print(f"[NOISE -> randn_local] randn_local - seed={seed}, shape={shape}")

    if ImageRNG._current_mode == "custom":
        print("[NOISE -> randn_local] Using quantum noise for local generation")
        return load_quantum_noise(shape, devices.device)

    # Original randn_local behavior for non-quantum mode
    if shared.opts.randn_source == "NV":
        print("[NOISE] Using NVIDIA Philox RNG generator")
        rng = rng_philox.Generator(seed)
        return torch.asarray(rng.randn(shape), device=devices.device)

    local_device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    print(f"[NOISE] Using PyTorch generator on {local_device}")
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)


"""
Input: x (tensor to match)
Output: noise tensor matching input shape
Used by: Not actively used
Purpose: Creates noise matching the shape of an input tensor
"""
def randn_like(x):
    """NOT USED: Generates noise matching shape of input tensor"""

    print(f"[NOISE -> randn_like] randn_like - shape={x.shape}")

    if shared.opts.randn_source == "NV":
        print("[NOISE] Using NVIDIA Philox RNG generator")
        return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)

    if shared.opts.randn_source == "CPU" or x.device.type == 'mps':
        print("[NOISE] Using PyTorch CPU generator")
        return torch.randn_like(x, device=devices.cpu).to(x.device)

    print(f"[NOISE] Using PyTorch generator on {x.device}")
    return torch.randn_like(x)


"""
HELPER FUNCTIONS USED BY DEFAULT BY STABLE DIFFUSION
"""

"""
Input: seed (int)
Output: none
Used by: randn()
Purpose: Sets up the RNG generator with a specific seed
"""
def manual_seed(seed):
    """USED INTERNALLY: Called by randn() to set up the generator"""

    if shared.opts.randn_source == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


"""
Input: seed (int)
Output: RNG generator object
Used by: ImageRNG.__init__()
Purpose: Creates RNG generator for either NVIDIA or PyTorch
"""
def create_generator(seed):
    """USED IN INITIALIZATION: Creates generators for ImageRNG"""
    if shared.opts.randn_source == "NV":
        print(f"[NOISE] Creating new NVIDIA Philox RNG generator with seed {seed}")
        return rng_philox.Generator(seed)

    device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator


"""
Input: val (float), low (tensor), high (tensor)
Output: interpolated tensor
Used by: ImageRNG.first() for subseed mixing
Purpose: Spherical interpolation between two noise tensors
"""
def slerp(val, low, high):
    """USED WHEN MIXING NOISES: Interpolates between two noise tensors"""
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


"""
CUSTOM FUNCTIONS FOR QUANTUM NOISE
"""


"""
Input: seed (int), shape (tuple), generator (optional)
Output: quantum noise tensor
Used by: ImageRNG.first() when mode="custom"
Purpose: Loads quantum noise from file for first step with optional blending
"""
def randnCustom(seed, shape, generator=None):
    """USED IN FIRST STEP: Loads quantum noise from file"""
    print(f"[QUANTUM NOISE -> randnCustom] Loading quantum noise")
    blend_config = BLEND_SETTINGS["first_step"]
    
    try:
        # Get quantum noise - shape should be [C, H, W]
        quantum_noise = load_quantum_noise(shape, devices.device)
        
        # If we want pure quantum noise, return it directly
        if blend_config["blend_ratio"] == 0.0:
            print("[QUANTUM NOISE] Using pure quantum noise")
            return quantum_noise.unsqueeze(0) if len(shape) == 4 else quantum_noise  # Add batch dim if needed
            
        # Generate standard noise
        manual_seed(seed)
        if shared.opts.randn_source == "NV":
            standard_noise = torch.asarray((generator or nv_rng).randn(shape), device=devices.device)
        elif shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
            standard_noise = torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        else:
            standard_noise = torch.randn(shape, device=devices.device, generator=generator)
            
        # Blend the noises using specified mode
        blended_noise = blend_noise(quantum_noise, standard_noise, 
                                  blend_config["blend_ratio"], 
                                  blend_config["blend_mode"])
        print(f"[QUANTUM NOISE] Blended noise with ratio {blend_config['blend_ratio']} using {blend_config['blend_mode']} mode")
        return blended_noise
        
    except Exception as e:
        print(f"[QUANTUM NOISE] Error loading quantum noise: {str(e)}, falling back to standard noise")
        return randn(seed, shape, generator)


"""
Input: shape (tuple), generator, blend_ratio (float), blend_mode (str)
Output: quantum noise tensor
Used by: ImageRNG.next() when mode="custom"
Purpose: Creates offset version of quantum noise for subsequent steps with optional blending
"""
def randn_without_seedCustom(shape, generator):
    """USED IN STEPS 2-20: Creates offset version of quantum noise"""
    print(f"[QUANTUM NOISE -> randn_without_seedCustom] Creating quantum noise")
    blend_config = BLEND_SETTINGS["subsequent_steps"]
    
    try:
        # Load the full quantum noise
        saved_noise = load_raw_quantum_noise()
        if saved_noise is None:
            raise ValueError("No quantum noise data loaded")
        
        # Generate random batch index
        batch_size = saved_noise.shape[0]
        batch_idx = int(torch.randint(0, batch_size, (1,)).item())
        print(f"[QUANTUM NOISE] Using batch index: {batch_idx} of {batch_size}")
        
        # Select random batch and prepare noise
        saved_noise = saved_noise[batch_idx:batch_idx+1]  # Keep batch dimension
        quantum_noise = prepare_quantum_noise(saved_noise, shape, devices.device)

        # If we want pure quantum noise, return it directly
        if blend_config["blend_ratio"] == 0.0:
            print("[QUANTUM NOISE] Using pure quantum noise")
            return quantum_noise.unsqueeze(0) if len(shape) == 4 else quantum_noise  # Add batch dim if needed

        # Generate standard noise
        if shared.opts.randn_source == "NV":
            standard_noise = torch.asarray((generator).randn(shape), device=devices.device)
        elif shared.opts.randn_source == "CPU" or devices.device.type == 'mps':
            standard_noise = torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        else:
            standard_noise = torch.randn(shape, device=devices.device, generator=generator)

        # Blend the noises using specified mode
        blended_noise = blend_noise(quantum_noise, standard_noise, 
                                  blend_config["blend_ratio"], 
                                  blend_config["blend_mode"])
        print(f"[QUANTUM NOISE] Blended noise with ratio {blend_config['blend_ratio']} using {blend_config['blend_mode']} mode")
        return blended_noise
        
    except Exception as e:
        print(f"[QUANTUM NOISE] Error processing quantum noise: {str(e)}, falling back to standard noise")
        return randn_without_seed(shape, generator)


"""
MAIN FUNCTION FROM STABLE DIFFUSION
"""


class ImageRNG:
    """MAIN CLASS: Manages noise generation for the entire process"""
    _instance = None
    _current_mode = DEFAULT_RNG_MODE  # Class-level mode storage
    
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, 
                 seed_resize_from_h=0, seed_resize_from_w=0, 
                 mode=None):  # Make mode optional
        print(f"[IMAGE_RNG -> init] Initializing ImageRNG with shape={shape}, seeds={seeds}, mode={mode or ImageRNG._current_mode}")
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.generators = [create_generator(seed) for seed in seeds]
        self.is_first = True
        self.mode = mode if mode is not None else ImageRNG._current_mode  # Use class-level mode if none provided
        
        # Update singleton instance
        ImageRNG._instance = self
    
    @classmethod
    def set_mode(cls, mode):
        """Class method to update the current mode"""
        cls._current_mode = mode
        if cls._instance is not None:
            cls._instance.mode = mode
    
    def first(self):
        print("[IMAGE_RNG -> first()] Starting ImageRNG.first() - Generating initial noise")
        print(f"[IMAGE_RNG -> first()] MODE: {self.mode}")
        
        # Select the noise function based on mode
        noise_func = randnCustom if self.mode == "custom" else randn
        
        noise_shape = self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 else (self.shape[0], int(self.seed_resize_from_h) // 8, int(self.seed_resize_from_w // 8))

        xs = []
        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            # OPTIONAL: Generate subnoise if subseeds are specified
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                print(f"[IMAGE_RNG -> first()] OPTIONAL SUBNOISE: Generating with subseed={subseed}")
                subnoise = noise_func(subseed, noise_shape)
            else:
                print("[IMAGE_RNG -> first()] OPTIONAL SUBNOISE: Disabled")

            # Generate main noise
            if noise_shape != self.shape:
                print(f"[IMAGE_RNG -> first()] Generating resized noise with seed={seed}")
                noise = noise_func(seed, noise_shape)
            else:
                print(f"[IMAGE_RNG -> first()] Generating standard noise with seed={seed}")
                noise = noise_func(seed, self.shape, generator=generator)

            # OPTIONAL: Mix noises if needed
            if subnoise is not None:
                print("[IMAGE_RNG -> first()] OPTIONAL MIXING: Combining main noise with subnoise")
                noise = slerp(self.subseed_strength, noise, subnoise)

            # OPTIONAL: Handle resizing
            if noise_shape != self.shape:
                print("[IMAGE_RNG -> first()] OPTIONAL RESIZE: Applying resize operation")
                x = noise_func(seed, self.shape, generator=generator)
                # ... resizing code ...
                noise = x

            xs.append(noise)

        # OPTIONAL: Handle eta noise
        if eta_noise_seed_delta := (shared.opts.eta_noise_seed_delta or 0):
            print(f"[IMAGE_RNG -> first()] OPTIONAL ETA NOISE: Creating new generators with delta={eta_noise_seed_delta}")
            self.generators = [create_generator(seed + eta_noise_seed_delta) for seed in self.seeds]
        else:
            print("[IMAGE_RNG -> first()] OPTIONAL ETA NOISE: Disabled")

        return torch.stack(xs).to(shared.device)

    def next(self):
        print("[IMAGE_RNG -> next()] Starting ImageRNG.next()")
        if self.is_first:
            self.is_first = False
            noise = self.first()
            print(f"[IMAGE_RNG -> next()] First step noise stats: mean={noise.mean():.4f}, std={noise.std():.4f}")
            return noise

        print(f"[IMAGE_RNG -> next()] MODE: {self.mode}")
        
        # Select the appropriate function based on mode
        noise_func = randn_without_seedCustom if self.mode == "custom" else randn_without_seed
            
        xs = []
        for generator in self.generators:
            x = noise_func(self.shape, generator=generator)
            # print(f"[IMAGE_RNG -> next()] Generated noise for generator: mean={x.mean():.4f}, std={x.std():.4f}")
            xs.append(x)
        noise = torch.stack(xs).to(shared.device)
        print(f"[IMAGE_RNG -> next()] Next step noise stats: mean={noise.mean():.4f}, std={noise.std():.4f}")
        return noise


devices.randn = randn
devices.randn_local = randn_local
devices.randn_like = randn_like
devices.randn_without_seed = randn_without_seed
devices.manual_seed = manual_seed

def blend_noise(quantum_noise, standard_noise, blend_ratio, mode="normal"):
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