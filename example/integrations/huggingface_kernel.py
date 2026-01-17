from kernels import get_kernel


kernel = get_kernel("kernels-community/liger_kernels")
layer = getattr(kernel.layers, "LigerRMSNorm", None)()
print(f"type(layer) = {type(layer)}")
