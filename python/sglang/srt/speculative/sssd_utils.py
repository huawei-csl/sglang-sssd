from __future__ import annotations

import copy
import logging
import torch
import torch.nn.functional as F
import hashlib
import bisect
import sssd_speculator
import time

from typing import List, Tuple, Optional
from itertools import chain
from dataclasses import dataclass
from transformers import AutoTokenizer

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import (
    align_evict_mask_to_page_size,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
    assign_req_to_token_pool,
    create_accept_length_filter,
    filter_finished_cache_loc_kernel
)
from sglang.srt.utils import next_power_of_2
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.speculative.eagle_utils import EagleVerifyInput
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.utils import is_cuda, is_hip, next_power_of_2
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    global_server_args_dict,
)

logger = logging.getLogger(__name__)

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
elif is_hip():
    from sgl_kernel import verify_tree_greedy

POSSIBLE_SPEC_LENS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

_COMPUTE_BW_RATIO = None


def get_compute_bw_ratio():
    global _COMPUTE_BW_RATIO
    if _COMPUTE_BW_RATIO is None:
        _COMPUTE_BW_RATIO = flops_to_bandwidth_ratio()
    logger.info(f"COMPUTE BW RATIO: {_COMPUTE_BW_RATIO}")
    return _COMPUTE_BW_RATIO


def get_max_speclen_per_bs(bs):
    """The roofline model does not apply well to small batch sizes, when the free budget is a lot and speculating
    too much is not worth it. So we put some additional manual constraints obtained by experience."""
    if bs <= 1:
        return 32
    if bs >=2 and bs < 4:
        return 64 // bs
    if bs >= 4 and bs <= 6:
        return 96 // bs
    # From here you can start trusting the roofline estimate
    return 512 // bs


class RequestIdToIntegerConverter:
    """
    Maps arbitrary strings to free 32-bit signed integers (0 … 2_147_483_647).

    • Same string ⇒ same number until `release()` is called.
    • Linear probing finds a gap if the first hash slot is taken.
    • Raises RuntimeError only when every int32 value is exhausted.
    • No thread-safety (single-thread use assumed).

    This is needed to map request ids (unsigned int128) to the speculator request ids (int32).
    """

    _MAX_INT32 = 2**31 - 1

    def __init__(self) -> None:
        self._str2int: dict[str, int] = {}
        self._used: set[int] = set()

    def acquire(self, key: str) -> Tuple[int, bool]:
        """
        Return an available int32 for `key`, remembering the choice.
        """
        if key in self._str2int:          # already allocated
            return self._str2int[key], True

        start = self._hash32(key)
        n = start
        while n in self._used:
            n = (n + 1) & self._MAX_INT32
            if n == start:
                raise RuntimeError("All 32-bit ints are in use")

        self._str2int[key] = n
        self._used.add(n)
        return n, False

    def release(self, ref) -> None:
        """
        Free the int32 previously assigned to `key` (no-op if unknown), or directly the int
        """
        val = self._str2int.pop(ref, None)
        if val is not None:
            self._used.discard(val)

    def __contains__(self, key: str) -> bool:
        return key in self._str2int

    def __len__(self) -> int:
        return len(self._used)

    @staticmethod
    def _hash32(s: str) -> int:
        """
        Deterministic 32-bit hash of the string `s`.
        (MD5 → take the first 4 bytes, big-endian, mask to 31 bits.)
        """
        digest = hashlib.md5(s.encode('utf-8')).digest()
        return int.from_bytes(digest[:4], 'big') & RequestIdToIntegerConverter._MAX_INT32


def merge_masks_to_flat(
        spec_masks : List[torch.Tensor],   # CPU Bool  (sLᵢ, sLᵢ)
        seq_lens : torch.Tensor,         # 1-D Long  (batch,)
        device : torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Takes the speculate-square tree masks ((spec_len, spec_len)), adds the "prefill part" ((spec_len, seq_len) of 1s),
    and flattens over batch size, in an efficient way.
    """
    assert len(spec_masks) == len(seq_lens), "batch mismatch or empty batch"

    # On CPU
    spec_lens = torch.tensor([m.size(0) for m in spec_masks], dtype=torch.long)
    blk_sizes = spec_lens * (spec_lens + seq_lens.cpu())
    offsets = torch.cat((torch.zeros(1, dtype=torch.long),
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


def merge_masks_to_flat_with_padding(
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


class SSSDSpeculator:
    "Python interface between SGLang and the C++ speculator code."
    def __init__(self, server_args: ServerArgs, device, captured_batch_sizes: List[int], num_tokens_per_bs_map: dict):
        if sssd_speculator is None:
            raise ImportError("SSSD speculator is not installed. Please install speculator first.")

        self.max_query_length = 4
        self.max_speculate_len = server_args.speculative_num_draft_tokens

        self.request_converter = RequestIdToIntegerConverter()
        self.device = device
        # If the speculation is not adaptive, these values are fixes, otherwise they are updated at each iteration
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk
        # In this is true the previous 3 values are not used
        self.speculative_adaptive = server_args.speculative_adaptive
        self.captured_batch_sizes = captured_batch_sizes
        self.num_tokens_per_bs_map = num_tokens_per_bs_map

        self.tokenizer = AutoTokenizer.from_pretrained(server_args.tokenizer_path)
        pad_tok = self.tokenizer.pad_token_id
        self.pad_token_id = pad_tok if pad_tok is not None else self.tokenizer.eos_token_id

        self._init_speculator(server_args)

    def _init_speculator(self, server_args):
        datastore_path = server_args.speculative_draft_model_path
        datastore_path = "" if datastore_path is None else datastore_path
        logger.info("Loading the SSSD speculator...")
        start_time = time.time()
        self.speculator = sssd_speculator.Reader(
            index_file_path=datastore_path,
            vocab_size=self.tokenizer.vocab_size + 2,
            stop_token=-1 if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id,
            max_search_entries=100,
            prompt_branch_length=8,
            prompt_prefix_length=self.max_query_length,
            max_output_size=server_args.num_reserved_decode_tokens,
            live_datastore=False,
            update_interval_ms=60 * 1000,
            max_update_chunk_size=512 * 1024 * 1024,
            max_indices=8,
            max_batch_size=server_args.max_running_requests
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Finished loading the SSSD speculator. Time taken: {elapsed_time:.2f} seconds")

    def add_new_sequence(self, req: Req):
        speculator_req_id, already_present = self.request_converter.acquire(req.rid)
        if already_present:
            logger.warning(
                f"Request {req.rid} already added to the speculator with id {req.sssd_id}, not adding it again")
        else:
            self.speculator.put(req.origin_input_ids, seq_id=speculator_req_id)
            req.sssd_id = speculator_req_id

    def stream_put(self, new_tokens: List[int], req: Req):
        assert req.sssd_id is not None, f"Request {req.rid} has no prompt inserted for speculation."
        self.speculator.stream_put(new_tokens=new_tokens, seq_id=req.sssd_id)

    def get_speculate_params_adaptive(self, batch: ScheduleBatch):
        bs = batch.batch_size()

        index = bisect.bisect_left(self.captured_batch_sizes, bs)
        if index >= len(self.captured_batch_sizes):
            return 1, 0, 0  # Beyond captured graphs, should not happen. Don't speculate
        pad_bs = self.captured_batch_sizes[index]
        self.num_draft_tokens = self.num_tokens_per_bs_map[pad_bs]
        self.num_steps, self.topk = default_branch_func(self.num_draft_tokens)

        return self.num_draft_tokens, self.num_steps, self.topk

    def get_draft(self, batch: ScheduleBatch):
        # TODO: For now assume all speculate_lens are the same (SGLang doesn't support variable spec_len out of the box)
        prefixes = []
        speculator_req_ids = []
        for req in batch.reqs:
            if len(req.output_ids) >= self.max_query_length:
                prefixes.append(req.output_ids[-self.max_query_length:])
            else:
                tokens_from_prompt = self.max_query_length - len(req.output_ids)
                prefixes.append(req.origin_input_ids[-tokens_from_prompt:] + req.output_ids)
            
            speculator_req_ids.append(req.sssd_id)

        bs = batch.batch_size()
        speculate_lens = [self.num_draft_tokens] * bs
        branch_lens = [self.num_steps] * bs
        max_topks = [self.topk] * bs

        # print("Toks, steps, topk: ", self.num_draft_tokens, self.num_steps, self.topk)
        
        (
            candidates,
            position_ids,
            next_token_ids,
            next_sibling_ids,
            decoding_masks
        ) = self.speculator.get_candidates_sglang(
                prefixes=prefixes,
                decoding_lengths=speculate_lens,
                branch_lengths=branch_lens,
                max_topks=max_topks,
                seq_ids=speculator_req_ids
            )
        
        # Convert numpy masks to torch (shares same memory)
        decoding_masks = [torch.from_numpy(m) for m in decoding_masks]

        seq_lens = batch.seq_lens
        
        # If some request doesn't have all the candidates requested, pad it
        to_pad = [self.num_draft_tokens - l for l in [len(c) for c in candidates]]
        if sum(to_pad) > 0:
            self._pad_speculate_outputs(
                candidates,
                position_ids,
                next_token_ids,
                next_sibling_ids,
                to_pad
            )
            full_tree_mask = merge_masks_to_flat_with_padding(
                decoding_masks,
                seq_lens,
                to_pad,
                self.num_draft_tokens,
                self.device,
            )
        else:
            full_tree_mask = merge_masks_to_flat(decoding_masks, seq_lens, self.device)
        
        flattened_positions = torch.tensor(list(chain.from_iterable(position_ids)), device=self.device)
        spec_lens = torch.tensor([len(s) for s in position_ids], device=self.device)
        expanded_offsets = seq_lens.repeat_interleave(spec_lens)
        draft_token_positions = flattened_positions + expanded_offsets

        retrive_index = torch.arange(spec_lens.sum(), device=self.device).view(len(spec_lens), -1)

        retrieve_next_token = torch.tensor(next_token_ids, device=self.device)

        retrive_next_sibling = torch.tensor(next_sibling_ids, device=self.device)
        
        draft_tokens = torch.tensor(list(chain.from_iterable(candidates)), device=self.device)

        return full_tree_mask, draft_token_positions, retrive_index, \
              retrieve_next_token, retrive_next_sibling, draft_tokens
        

    def clear_seq_cache(self, req_id: str) -> None:
        speculator_req_id, already_present = self.request_converter.acquire(req_id)
        assert already_present, "Trying to remove a sequence that is not in the speculator"
        self.speculator.finish_sequence(seq_id=speculator_req_id)
        self.request_converter.release(req_id)

    def _pad_speculate_outputs(
        self,
        candidates: List[List[int]],
        position_ids: List[List[int]],
        next_token_ids: List[List[int]],
        next_sibling_ids: List[List[int]],
        to_pad: List[int],
    ):
        for idx, pad_size in enumerate(to_pad):
            if pad_size > 0:
                candidates[idx] += [self.pad_token_id] * pad_size
                position_ids[idx] += [-1] * pad_size
                next_token_ids[idx] += [-1] * pad_size
                next_sibling_ids[idx] += [-1] * pad_size


@dataclass
class SSSDVerifyInput(EagleVerifyInput):

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            return SSSDVerifyInput(
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrive_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device="cuda")
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device="cuda"
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device="cuda")

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrive_index are the indices of the requests that are kept.
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        if sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device="cuda",
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # Sample tokens
        if batch.sampling_info.is_all_greedy:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

            verify_tree_greedy(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            # apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            if not torch.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device="cuda"
            )

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device="cuda")
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device="cuda"
            )
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=global_server_args_dict[
                    "speculative_accept_threshold_single"
                ],
                threshold_acc=global_server_args_dict[
                    "speculative_accept_threshold_acc"
                ],
                deterministic=True,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        finished_requests = []
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    finished_requests.append(req.rid)
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                            )
                            raise e
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1

        if has_finished:
            # Recompute for all
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = accept_length[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        accepted_lenghts_plus_bonus = accept_length + 1   # with bonus token
        accepted_by_req_gpu = verified_id.split(accepted_lenghts_plus_bonus.tolist())   # tuple of bs tensors
        new_tokens_per_req = [t.cpu().tolist() for t in accepted_by_req_gpu]

        # Construct SSSDVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool[(bs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(bs),
                )
            else:
                batch.out_cache_loc = tgt_cache_loc

            batch.seq_lens.add_(accepted_lenghts_plus_bonus)

            accept_length_cpu = accept_length.tolist()

            return SSSDVerifyOutput(
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_cpu,
                accepted_indices=accept_index,
                new_accepted_tokens=new_tokens_per_req,
                finished_requests=finished_requests
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool[(bs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(bs),
                )
                batch.seq_lens.add_(accepted_lenghts_plus_bonus)

            accept_length_cpu = accept_length.tolist()
            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_accept_length_cpu = [
                    accept_length_cpu[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

            return SSSDVerifyOutput(
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_cpu,
                accepted_indices=accept_index,
                new_accepted_tokens=new_tokens_per_req,
                finished_requests=finished_requests
            )
    
    def filter_batch(self, spec_info: SSSDVerifyInput):
        # Needed for DraftInput, not for VerifyInput, so doesn't hold for sssd (but still gets triggered)
        pass

    def merge_batch(self, spec_info: SSSDVerifyInput):
        # Needed for DraftInput, not for VerifyInput, so doesn't hold for sssd (but still gets triggered)
        pass

@dataclass
class SSSDVerifyOutput:
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: torch.Tensor
    # Tokens to add to speculator
    new_accepted_tokens: List[List[int]]
    # Finished requests, can be removed from speculator
    finished_requests: List[str]

### UTILS ###
    
def default_branch_func(speculate_len: int):
        if speculate_len <= 5:
            branch_length = speculate_len - 1
        elif speculate_len <= 8:
            branch_length = 5
        elif speculate_len <= 32:
            branch_length = 6
        elif speculate_len <= 48:
            branch_length = 8
        else:
            branch_length = 10

        return branch_length, min(5, speculate_len-1)


def flops_to_bandwidth_ratio(copy_bytes: int = 1 * 1024 * 1024 * 1024,
                             mm_size: int = 8192,
                             repeats: int = 100) -> float:
    """
    Estimate GPU balance point (GFLOPs per GB/s) using roofline-style benchmarking.

    - Bandwidth: measured with streaming AXPY-like kernel (2 reads + 1 write).
    - FLOPs: measured with large GEMM on tensor cores (BF16).
    - Ratio = GFLOPs / GB/s, i.e. arithmetic intensity required to be compute-bound.

    Args:
        copy_bytes (int): Size of the buffer for bandwidth test (default: 1 GiB).
        mm_size (int): Matrix size for GEMM test (default: 8192).
        repeats (int): Number of iterations to average (default: 100).

    Returns:
        float: FLOPs-to-bandwidth ratio (GFLOPs per GB/s).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available — run on a GPU-equipped system.")

    # Bandwidth measurement
    elems = copy_bytes // 2  # bf16 = 2 bytes
    a = torch.randn(elems, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)

    # warm-up
    for _ in range(3):
        torch.add(a, b, out=b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(repeats):
        torch.add(a, b, out=b)  # 2 reads + 1 write
    end.record()
    torch.cuda.synchronize()

    avg_ms_bw = start.elapsed_time(end) / repeats
    bandwidth_gbps = (3 * copy_bytes * 1e-9) / (avg_ms_bw * 1e-3)

    # FLOPs measurement
    A = torch.randn(mm_size, mm_size, device="cuda", dtype=torch.bfloat16)
    B = torch.randn_like(A)

    for _ in range(3):
        torch.matmul(A, B)
    torch.cuda.synchronize()

    start.record()
    for _ in range(repeats):
        torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()

    avg_ms_flops = start.elapsed_time(end) / repeats
    flops_gflops = (2.0 * mm_size ** 3 / 1e9) / (avg_ms_flops * 1e-3)

    return flops_gflops / bandwidth_gbps
