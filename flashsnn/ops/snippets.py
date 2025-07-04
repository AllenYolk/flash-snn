import triton
import triton.language as tl


@triton.jit
def convert_and_store(pointer, value, boundary_check):
    value = value.to(pointer.dtype.element_ty.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
