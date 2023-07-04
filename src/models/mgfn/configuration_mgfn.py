from transformers.configuration_utils import PretrainedConfig


class MGFNConfig(PretrainedConfig):
    def __init__(
        self,
        classes=0,
        dims=(64, 128, 1024),
        depths=(3, 3, 2),
        mgfn_types=("gb", "fb", "fb"),
        lokernel=5,
        channels=2048,
        ff_repe=4,
        dim_head=64,
        local_aggr_kernel=5,
        dropout=0.0,
        attention_dropout=0.0,
        dropout_rate=0.7,
        mag_ratio=0.1,
        k=3,
    ):
        super().__init__()
        self.classes = classes
        self.dims = dims
        self.depths = depths
        self.mgfn_types = mgfn_types
        self.lokernel = lokernel
        self.channels = channels
        self.ff_repe = ff_repe
        self.dim_head = dim_head
        self.local_aggr_kernel = local_aggr_kernel
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.dropout_rate = dropout_rate
        self.mag_ratio = mag_ratio
        self.k = k
