import torch
from typing import List

torch.set_printoptions(
    threshold=float('inf'),   # Don't truncate elements
    linewidth=200,            # Number of characters per line
    edgeitems=None,           # Print all rows/cols instead of edge only
    precision=4,              # Optional: decimal precision
    sci_mode=False            # Optional: disable scientific notation
)

# FIXED SPECULATION LENGTH, NAIVE

def rebuild_full_block(spec_mask, seq_len, speculate_len):
    """
    spec_mask : (speculate_len , speculate_len)  Bool tensor
    returns    : (speculate_len , seq_len + speculate_len) Bool tensor
    """
    device = 'cuda'
    full_cols = seq_len + speculate_len

    block = torch.ones((speculate_len, full_cols),
                        dtype=spec_mask.dtype, device=device)

    # Copy the interesting part:
    block[:, seq_len:] = spec_mask
    # everything else already zero

    return block

def merge_masks_to_flat(spec_masks, seq_lens):
    device = 'cuda'
    bs  = len(spec_masks)
    sl  = seq_lens.to(device)
    spec_len = spec_masks[0].size(0)

    # --- build every block ---
    blocks = [rebuild_full_block(m, sl[i].item(), spec_len)
              for i, m in enumerate(spec_masks)]

    # --- compute the flat tensor size and allocate ---
    total_bits = (spec_len*spec_len)*bs \
               + spec_len*sl.sum().item()
    out = torch.zeros(total_bits, dtype=torch.bool, device=device)

    # --- paste every block in its proper slice ---
    running_prefix = 0          # Σ_{i<b} seq_len[i]
    for b, blk in enumerate(blocks):
        offset = spec_len*spec_len * b \
               + spec_len * running_prefix
        out[offset : offset + blk.numel()] = blk.flatten()
        running_prefix += sl[b].item()

    return out

# VARIABLE SPECULATION LENGTH, NAIVE

def rebuild_full_block_variable(spec_mask: torch.Tensor,
                       seq_len: int) -> torch.Tensor:
    """
    spec_mask : (spec_len, spec_len)  Bool  - assumed to be on **CPU**
    seq_len   : number of “prompt” tokens that precede the speculative block

    returns   : (spec_len, seq_len + spec_len) Bool tensor
                - lives on the *same* device as spec_mask
    """
    spec_len = spec_mask.size(0)                        # <- now per-sample
    device   = "cuda"

    block = torch.ones((spec_len, seq_len + spec_len),
                       dtype=spec_mask.dtype,
                       device=device)                   # left half = 1s
    block[:, seq_len:] = spec_mask                      # right half

    return block


def merge_masks_to_flat_variable(
        spec_masks : List[torch.Tensor],   # list of (sLᵢ, sLᵢ)  Bool  (CPU)
        seq_lens   : torch.Tensor,             # 1-D long  (batch,)
        device     : torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Concatenate all per-sample blocks row-wise into a single 1-D Bool tensor
    that lives on `device`.

    spec_len may differ for every sample.
    """
    assert len(spec_masks) == len(seq_lens), "batch mismatch"
    B = len(spec_masks)

    # ----- figure out how big the output must be ---------------------------
    spec_lens  = torch.tensor([m.size(0) for m in spec_masks], dtype=torch.long)
    blk_sizes  = spec_lens * (spec_lens + seq_lens.cpu())    # elements per sample
    total_bits = int(blk_sizes.sum())

    out = torch.zeros(total_bits, dtype=torch.bool, device=device)

    # ----- fill it ----------------------------------------------------------
    running_offset = 0
    for i in range(B):
        # 1. build the (spec_lenᵢ , seq_lenᵢ + spec_lenᵢ) block on CPU
        blk = rebuild_full_block_variable(spec_masks[i], int(seq_lens[i]))

        # 2. copy it to the output slice on the GPU
        blk_gpu = blk.to(device)                    # regular .to(); no pinning
        out[running_offset : running_offset + blk_gpu.numel()] = blk_gpu.flatten()

        running_offset += blk_gpu.numel()           # advance

    return out

# SAME AS ABOVE, FASTER

def merge_masks_to_flat_variable_fast(
        spec_masks : List[torch.Tensor],   # CPU Bool  (sLᵢ, sLᵢ)
        seq_lens   : torch.Tensor,         # 1-D Long  (batch,)
        device     : torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Like merge_masks_to_flat_variable(), but
      * no temporaries on the GPU
      * only the small spec_masks cross PCIe
      * every copy is non-blocking and overlaps with the in-GPU fill
    """
    assert len(spec_masks) == len(seq_lens), "batch mismatch or empty batch"

    # On CPU
    spec_lens = torch.tensor([m.size(0) for m in spec_masks], dtype=torch.long)
    blk_sizes = spec_lens * (spec_lens + seq_lens.cpu())
    offsets   = torch.cat((torch.zeros(1, dtype=torch.long),
                           torch.cumsum(blk_sizes[:-1], 0)))
    total_bits = int(blk_sizes.sum().item())

    # On GPU
    out = torch.empty(total_bits, dtype=torch.bool, device=device)
    out.fill_(True)

    stream = torch.cuda.current_stream(device)
    with torch.cuda.stream(stream):        # make stream explicit, just in case
        for cpu_mask, seq_len, spec_len, start in zip(
                spec_masks,
                seq_lens.tolist(),
                spec_lens.tolist(),
                offsets.tolist()):

            # Slice view into the final buffer
            blk = out[start : start + spec_len * (seq_len + spec_len)] \
                    .view(spec_len, seq_len + spec_len)

            # Right half ← speculative mask (async H->D DMA)
            pinned = cpu_mask if cpu_mask.is_pinned() else cpu_mask.pin_memory()
            blk[:, seq_len:].copy_(pinned, non_blocking=True)

    # no synchronise here – let the caller decide
    return out

# VARIABLE SPECULATION LENGTH, PARALLEL

def merge_masks_to_flat_variable_parallel(
        spec_masks : List[torch.Tensor],   # (sLᵢ, sLᵢ)  CPU Bool
        seq_lens   : torch.Tensor,         # (batch,)   Long -- can live on CPU
        device     : torch.device = torch.device('cuda')
) -> torch.Tensor:
    """
    Concatenate all per-sample blocks

        [  ones(spec_lenᵢ, seq_lenᵢ) | spec_maskᵢ ]

    into one 1-D Bool tensor that already lives on `device`.
    All GPU work is fully asynchronous; only the small spec_masks travel
    over PCIe (pinned, non-blocking).  One CUDA stream per sample lets the
    fills and copies overlap.
    """
    assert len(spec_masks) == len(seq_lens), "batch mismatch"
    B = len(spec_masks)
    assert B > 0, "empty batch"

    # Compute sizes/offsets on the CPU
    spec_lens = torch.tensor([m.size(0) for m in spec_masks], dtype=torch.long)
    blk_sizes = spec_lens * (spec_lens + seq_lens.cpu())
    offsets   = torch.cat([torch.zeros(1, dtype=torch.long),
                           torch.cumsum(blk_sizes[:-1], 0)])
    total_bits = int(blk_sizes.sum().item())

    # Allocate the final buffer on the target GPU
    out = torch.empty(total_bits, dtype=torch.bool, device=device)
    out.fill_(True)

    # 3) Launch one CUDA stream per sample
    #    * left half fill_(1)  runs on the GPU
    #    * right half copy    uses the same stream so the order is correct

    streams = [torch.cuda.Stream(device=device) for _ in range(B)]

    for i, (cpu_mask, seq_len, spec_len, start) in enumerate(
            zip(spec_masks,
                seq_lens.tolist(),
                spec_lens.tolist(),
                offsets.tolist())):

        # Pin once; no need to clone.  Non-blocking copy requires pinned host mem
        pinned = cpu_mask.pin_memory()

        blk_elements = spec_len * (seq_len + spec_len)
        blk_view = out[start : start + blk_elements].view(spec_len, seq_len + spec_len)

        s = streams[i]                      # dedicated stream
        with torch.cuda.stream(s):
            blk_view[:, seq_len:].copy_(pinned, non_blocking=True)

    torch.cuda.synchronize(device)
    return out

def merge_masks_to_flat_variable_fast_with_padding(
        spec_masks : List[torch.Tensor],   # CPU Bool  (sLᵢ, sLᵢ)
        seq_lens : torch.Tensor,         # 1-D Long  (batch,)
        pad_dims : List[int],
        max_spec_len: int,
        device : torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Takes the speculate-square tree masks ((spec_len, spec_len)), adds the "prefill part" ((spec_len, seq_len) of 1s),
    and flattens over batch size, in an efficient way.
    """
    assert len(spec_masks) == len(seq_lens), "batch mismatch or empty batch"

    # On CPU
    real_spec_lens = torch.tensor([m.size(0) for m in spec_masks], dtype=torch.long)
    spec_lens = torch.tensor([max_spec_len]*len(spec_masks), dtype=torch.long)
    blk_sizes = spec_lens * (spec_lens + seq_lens.cpu())
    offsets = torch.cat((torch.zeros(1, dtype=torch.long),
                           torch.cumsum(blk_sizes[:-1], 0)))
    total_bits = int(blk_sizes.sum().item())

    # On GPU
    out = torch.empty(total_bits, dtype=torch.bool, device=device)
    out.fill_(True)

    stream = torch.cuda.current_stream(device)
    with torch.cuda.stream(stream):        # make stream explicit, just in case
        for cpu_mask, seq_len, spec_len, real_spec_len, pad_size, start in zip(
                spec_masks,
                seq_lens.tolist(),
                spec_lens.tolist(),
                real_spec_lens.tolist(),
                pad_dims,
                offsets.tolist()):

            # Slice view into the final buffer
            blk = out[start : start + spec_len * (seq_len + spec_len)] \
                    .view(spec_len, seq_len + spec_len)

            # Right half ← speculative mask (async H->D DMA)
            pinned = cpu_mask if cpu_mask.is_pinned() else cpu_mask.pin_memory()
            if pad_size == 0:
                blk[:, seq_len:].copy_(pinned, non_blocking=True)
            else:
                blk[:real_spec_len, seq_len:seq_len+real_spec_len].copy_(pinned, non_blocking=True)
                # Padding
                blk[real_spec_len:, :].fill_(False)
                blk[:real_spec_len, seq_len+real_spec_len:].fill_(False)                

    # no synchronise here – let the caller decide
    return out


mask1 = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]).bool()

mask2 = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]]).bool()


full_mask = torch.tensor([
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]).bool()

# # Opposite operation: from flattened mask to single masks
# def _extract_single_spec_mask(
#             tree_mask_flat      : torch.BoolTensor,
#             seq_lens            : torch.LongTensor,    # (bs,)
#             batch_idx           : int,
#             draft_token_num     : int,
#             speculate_len       : int
#     ):
#         """
#         Returns the boolean mask for one batch element in the shape
#         (speculate_len , seq_len + speculate_len).
#         """
#         # ---- reproduce the index arithmetic from build_tree_efficient ----
#         seq_prefix = seq_lens[:batch_idx].sum().item()
#         offset     = draft_token_num * draft_token_num * batch_idx
#         offset    += draft_token_num * seq_prefix

#         seq_len   = seq_lens[batch_idx].item()
#         total_len = draft_token_num * (seq_len + draft_token_num)

#         # ---- reshape and slice ----
#         full = (tree_mask_flat            # (total_len,)  →  (draft_token_num , seq_len + draft_token_num)
#                 .narrow(0, offset, total_len)
#                 .view(draft_token_num, seq_len + draft_token_num))

#         # drop root row (row-0) and the “extra” draft columns
#         # return full[1:1 + speculate_len, :seq_len + speculate_len].contiguous()
#         return full.contiguous()

# bs = 2                 # batch size
# draft_token_num = 16
# speculate_len = draft_token_num        # == num_speculative_tokens

# for b in range(bs):
#     single_mask = _extract_single_spec_mask(
#         full_mask, torch.tensor([19, 18]), b,
#         draft_token_num, speculate_len
#     )
#     # Move to CPU just for nicer printing if you are on CUDA
#     print(f"[batch {b}] individual mask "
#         f"shape = {single_mask.shape} (rows = speculate_len, cols = seq_len+spec_len)\n",
#         single_mask.int().cpu())
    
# print("Full mask shape: ", full_mask.shape)


# TEST CORRECTNESS

# spec_mask_1 = mask1[:, 19:]
# spec_mask_2 = mask2[:, 18:]

# masks = [spec_mask_1, spec_mask_2]
# seq_lens = torch.tensor([19, 18])
# res = merge_masks_to_flat(masks, seq_lens)
# res_var = merge_masks_to_flat_variable(masks, seq_lens)
# res_par = merge_masks_to_flat_variable_parallel(masks, seq_lens)

# assert torch.equal(res, full_mask.to("cuda")), f"{res} {full_mask}"
# assert torch.equal(res_var, full_mask.to("cuda")), f"{res} {res_var}"
# assert torch.equal(res_par, full_mask.to("cuda")), f"{res} {res_par}"


spec_mask_1 = mask1[:, 19:]
spec_mask_2 = mask2[:12, 18:30]

print(spec_mask_1.int())
print(spec_mask_2.int())

masks = [spec_mask_1, spec_mask_2]
seq_lens = torch.tensor([8, 9])
res = merge_masks_to_flat_variable_fast_with_padding(masks, seq_lens, [1, 5], 17)

print(res.int())

# import timeit

# n_iters = 100  # Run each function 100 times

# masks = [spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2]
# positions = torch.tensor([10240, 1200, 800, 1000, 2000, 194, 10110, 75, 8000, 1000, 2000, 194, 1024, 1200, 80400, 789])

# # masks = [spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2, spec_mask_1, spec_mask_2]
# # positions = torch.tensor([1024, 120, 800, 100, 200, 194, 1010, 75, 800, 100, 200, 194, 1024, 120, 804, 789])

# res = merge_masks_to_flat(masks, positions)
# res_var = merge_masks_to_flat_variable(masks, positions)
# res_var_fast = merge_masks_to_flat_variable_fast(masks, positions)
# res_par = merge_masks_to_flat_variable_parallel(masks, positions)

# assert torch.equal(res_var, res.to("cuda")), f"{res.int()} {res_var.int()}"
# assert torch.equal(res_var_fast, res.to("cuda")), f"{res.int()} {res_var_fast.int()}"
# assert torch.equal(res_par, res.to("cuda")), f"{res.int()} {res_par.int()}"

# def timed_flat():
#     torch.cuda.synchronize()      # 1) make sure the GPU is idle
#     merge_masks_to_flat(masks, positions)
#     torch.cuda.synchronize()

# def timed_var():
#     torch.cuda.synchronize()
#     merge_masks_to_flat_variable(masks, positions)
#     torch.cuda.synchronize()

# def timed_var_fast():
#     torch.cuda.synchronize()
#     merge_masks_to_flat_variable_fast(masks, positions)
#     torch.cuda.synchronize()

# def timed_var_parallel():
#     torch.cuda.synchronize()
#     merge_masks_to_flat_variable_parallel(masks, positions)
#     torch.cuda.synchronize()     

# time_flat = timeit.timeit(lambda: timed_flat(), number=n_iters)
# avg_flat = (time_flat / n_iters) * 1e6
# print(f"merge_masks_to_flat: {avg_flat:.2f} µs per iteration")

# time_var = timeit.timeit(lambda: timed_var(), number=n_iters)
# avg_var = (time_var / n_iters) * 1e6
# print(f"merge_masks_to_flat_variable: {avg_var:.2f} µs per iteration")

# time_var = timeit.timeit(lambda: timed_var_fast(), number=n_iters)
# avg_var = (time_var / n_iters) * 1e6
# print(f"merge_masks_to_flat_variable: {avg_var:.2f} µs per iteration")

# time_par = timeit.timeit(lambda: timed_var_parallel(), number=n_iters)
# avg_par = (time_par / n_iters) * 1e6
# print(f"merge_masks_to_flat_variable_parallel: {avg_par:.2f} µs per iteration")


1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0