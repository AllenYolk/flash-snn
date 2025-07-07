import triton
import triton.language as tl


@triton.jit
def convert_and_store(pointer, value, boundary_check):
    """For block pointers created by tl.make_block_pointer(), implicit type
    casting is not supported when called tl.store(). This function is a wrapper
    that first converts the value to the pointer's element type, and then
    calls tl.store(). 

    This function is mainly used by flexsn to ensure type validity.

    For other cases, we suggest passing dtypes as constexpr arguments to the 
    kernels rather than using this wrapper. In this way, Triton will compile 
    multiple versions of the kernel for different dtypes and skip unnecessary 
    type conversions!
    """
    value = value.to(pointer.dtype.element_ty.element_ty)
    tl.store(pointer, value, boundary_check=boundary_check)
