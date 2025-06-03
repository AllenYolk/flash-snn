import torch
import triton.language as tl

type_dict = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
}

# check bfloat16 compatibility
dc = torch.cuda.get_device_capability()
if dc[0] < 8 or not hasattr(tl, "bfloat16"):
    print(
        "Triton kernel with bfloat16 is not supported on devices "
        "with compute capability < 8.0. "
        f"Your device's capability is: {dc}."
    )
    TRITON_BFLOAT16_AVAILABLE = False
else:
    TRITON_BFLOAT16_AVAILABLE = True
    type_dict[torch.bfloat16] = tl.bfloat16

# check float8_e4m3fn compatibility
TORCH_FLOAT8E4M3FN_AVAILABLE = hasattr(torch, "float8_e4m3fn")
if float(f"{dc[0]}.{dc[1]}") < 8.9 or not hasattr(tl, "float8e4nv"):
    print(
        "Triton kernel with float8e4nv (float8_e4m3fn) is not supported on "
        "devices with compute capability < 8.9. "
        f"Your devices's capability is: {dc}."
    )
    TRITON_FLOAT8E4NV_AVAILABLE = False
else:
    TRITON_FLOAT8E4NV_AVAILABLE = True
    if TORCH_FLOAT8E4M3FN_AVAILABLE:
        type_dict[torch.float8_e4m3fn] = tl.float8e4nv
