import logging

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.model_free_worker import ModelFreeWorker
from sglang.srt.speculative.pld_utils import PLDSpeculator

logger = logging.getLogger(__name__)


class PLDWorker(ModelFreeWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(server_args, gpu_id, tp_rank, target_worker)

        if self.tp_rank == 0:
            self.speculator = PLDSpeculator(
                server_args,
                self.device,
                self.captured_batch_sizes,
                self.num_tokens_per_bs_map,
                self.tp_rank,
                self.tp_group,
            )
