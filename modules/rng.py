"""
Todo:
- blend with normal generated noise
- ui for custom noise
- control steps
"""

import torch
import os

from modules import devices, rng_philox, shared

_NOISE_PATH = r"C:\Users\mail\Github\quantum-noise\quantum-noise-transform\output\qn-full-latent\latent_qn-sample-high.pt"
NOISE_FILE = os.path.join(*_NOISE_PATH.split("\\"))

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

def power_transform(noise, power=1.0):
    """Changes the "sharpness" of the noise"""
    return torch.sign(noise) * torch.abs(noise) ** power

def mix_with_gaussian(noise, mix_ratio=0.5):
    """Blends quantum noise with traditional Gaussian noise"""
    gaussian = torch.randn_like(noise)
    return (1 - mix_ratio) * noise + mix_ratio * gaussian

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

def load_raw_quantum_noise():
    """Load the raw quantum noise from file"""
    print(f"[QUANTUM NOISE] Using noise file: {NOISE_FILE}")
    saved_noise = torch.load(NOISE_FILE)
    print(f"[QUANTUM NOISE] Loaded noise shape: {saved_noise.shape}")
    return saved_noise

def prepare_quantum_noise(saved_noise, shape, device, 
                        start_idx=None,     
                        scale_y=1.0,        
                        scale_x=1.0,        
                        normalization="gaussian",
                        norm_strength=0.0,  
                        power=1.0,          
                        gaussian_mix=0.0,    
                        high_pass=0.0,      
                        low_pass=1.0,       
                        num_scales=1):      
    """Enhanced quantum noise preparation with all modifications"""
    
    # Reshape saved_noise to ensure it's 3D (channels, height, width)
    if len(saved_noise.shape) == 4:
        saved_noise = saved_noise.squeeze(0)  # Remove batch dimension if present
    
    # Get basic noise sample
    if saved_noise.shape != (4, 128, 128):
        # Resize if dimensions don't match
        saved_noise = torch.nn.functional.interpolate(
            saved_noise.unsqueeze(0),  # Add batch dimension for interpolation
            size=(128, 128),
            mode='bilinear'
        ).squeeze(0)  # Remove batch dimension
    
    noise = saved_noise * scale_y
    
    # Apply modifications in sequence
    noise = normalize_noise(noise, method=normalization, strength=norm_strength)
    
    # 2. Power transform
    if power != 1.0:
        noise = power_transform(noise, power)
    
    # 3. Frequency modifications
    if high_pass > 0 or low_pass < 1.0:
        noise = frequency_modify(noise, high_pass, low_pass)
    
    # 4. Multi-scale mixing
    if num_scales > 1:
        noise = create_multiscale_noise(noise, num_scales)
    
    # 5. Mix with Gaussian
    if gaussian_mix > 0:
        noise = mix_with_gaussian(noise, gaussian_mix)
    
    # Prepare for output
    if len(shape) == 4:
        noise = noise.unsqueeze(0)
        if shape[0] > 1:
            noise = noise.expand(shape[0], -1, -1, -1)
    
    return noise.to(device)

def load_quantum_noise(shape, device):
    """Load and prepare quantum noise from file"""
    raw_noise = load_raw_quantum_noise()
    return prepare_quantum_noise(raw_noise, shape, device)

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


def randnCustom(seed, shape, generator=None):
    """USED IN FIRST STEP: Loads quantum noise from file"""
    print(f"[QUANTUM NOISE -> randnCustom] Loading quantum noise")
    
    try:
        quantum_noise = load_quantum_noise(shape, devices.device)
        print(f"[QUANTUM NOISE] Successfully loaded quantum noise from file")
        return quantum_noise
    except Exception as e:
        print(f"[QUANTUM NOISE] Error loading quantum noise: {str(e)}, falling back to standard noise")
        # Fallback to standard noise generation if loading fails
        return randn(seed, shape, generator)



def randn_local(seed, shape):
    """NOT USED: Alternative noise generator that doesn't change global state"""

    print(f"[NOISE -> randn_local] randn_local - seed={seed}, shape={shape}")

    if shared.opts.randn_source == "NV":
        print("[NOISE] Using NVIDIA Philox RNG generator")
        rng = rng_philox.Generator(seed)
        return torch.asarray(rng.randn(shape), device=devices.device)

    local_device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    print(f"[NOISE] Using PyTorch generator on {local_device}")
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)


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


def randn_without_seedCustom(shape, generator=None, mode="normal"):
    """USED IN STEPS 2-20: Creates offset version of quantum noise"""
    print(f"[QUANTUM NOISE -> randn_without_seedCustom] Creating {mode} quantum noise")
    
    # Load the full quantum noise
    saved_noise = load_raw_quantum_noise()
    
    if mode == "fixed":
        # Always use the same starting position
        x_offset = 0
        y_offset = 0
    else:
        # Generate random offsets for x and y
        if generator is not None:
            rand_tensor = torch.empty(2).to(generator.device)
            rand_tensor.random_(generator=generator)
            x_offset = int(rand_tensor[0].item() * (128 - shape[-1]))
            y_offset = int(rand_tensor[1].item() * (128 - shape[-2]))
        else:
            x_offset = torch.randint(0, 128 - shape[-1], (1,)).item()
            y_offset = torch.randint(0, 128 - shape[-2], (1,)).item()
    
    # Take a slice starting from the random offset
    saved_noise = saved_noise[..., y_offset:y_offset + shape[-2], x_offset:x_offset + shape[-1]]
    
    return prepare_quantum_noise(saved_noise, shape, devices.device)


def manual_seed(seed):
    """USED INTERNALLY: Called by randn() to set up the generator"""

    if shared.opts.randn_source == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


def create_generator(seed):
    """USED IN INITIALIZATION: Creates generators for ImageRNG"""
    if shared.opts.randn_source == "NV":
        print(f"[NOISE] Creating new NVIDIA Philox RNG generator with seed {seed}")
        return rng_philox.Generator(seed)

    device = devices.cpu if shared.opts.randn_source == "CPU" or devices.device.type == 'mps' else devices.device
    generator = torch.Generator(device).manual_seed(int(seed))
    return generator


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
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


class ImageRNG:
    """MAIN CLASS: Manages noise generation for the entire process"""
    
    def __init__(self, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, mode="custom"):
        print(f"[IMAGE_RNG -> init] Initializing ImageRNG with shape={shape}, seeds={seeds}, mode={mode}")
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w
        self.generators = [create_generator(seed) for seed in seeds]
        self.is_first = True
        self.mode = mode

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