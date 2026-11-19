import torch


def enable_fast_matmul() -> None:
    """The cuDNN and matmul settings training uses; evaluation sets them too so that it computes the same way."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
