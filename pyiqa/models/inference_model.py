import torch
import torch.nn.functional as F

from collections import OrderedDict
from pyiqa.default_model_configs import DEFAULT_CONFIGS
# from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs import build_network
from pyiqa.utils.img_util import imread2tensor

from pyiqa.losses.loss_util import weight_reduce_loss
from pyiqa.archs.arch_util import load_pretrained_network

class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            seed=123,
            check_input_range=True,
            enable_mixed_precision=False,
            memory_efficient=True,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        self.score_range = DEFAULT_CONFIGS[metric_name].get('score_range', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Optimize CUDNN settings for performance
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Enable for better performance
            torch.backends.cudnn.deterministic = False  # Disable for speed when not needed for reproducibility
        
        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction
        # disable input range check when used as loss
        self.check_input_range = check_input_range if not as_loss else False
        
        # Performance optimization options
        self.enable_mixed_precision = enable_mixed_precision
        self.memory_efficient = memory_efficient

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        self.net = build_network(net_opts)
        self.net = self.net.to(self.device)
        
        # Set model to eval mode and optimize for inference
        self.net.eval()
        
        # Enable optimizations for inference
        if hasattr(self.net, 'fuse_model'):
            self.net.fuse_model()  # Fuse conv-bn if available
        
        # JIT compile for supported models when not in training mode
        if not as_loss and hasattr(torch.jit, 'optimize_for_inference'):
            try:
                self.net = torch.jit.optimize_for_inference(self.net)
            except Exception:
                pass  # Fall back to regular model if JIT optimization fails

        self.seed = seed

        # Use a single parameter to track device instead of dummy tensor
        self.register_buffer('_device_tracker', torch.empty(0))
    
    def load_weights(self, weights_path, weight_keys='params'):
        load_pretrained_network(self.net, weights_path, weight_keys=weight_keys, device=self.device)
    
    def is_valid_input(self, x):
        if x is not None:
            assert isinstance(x, torch.Tensor), 'Input must be a torch.Tensor'
            assert x.dim() == 4, 'Input must be 4D tensor (B, C, H, W)'
            assert x.shape[1] in [1, 3], 'Input must be RGB or gray image'
        
            if self.check_input_range:
                assert x.min() >= 0 and x.max() <= 1, f'Input must be normalized to [0, 1], but got min={x.min():.4f}, max={x.max():.4f}'
    
    def _preprocess_input(self, target, ref=None):
        """Optimized input preprocessing with memory efficiency."""
        device = self._device_tracker.device

        if not torch.is_tensor(target):
            target = imread2tensor(target, rgb=True)
            target = target.unsqueeze(0)
            if self.metric_mode == 'FR':
                assert ref is not None, 'Please specify reference image for Full Reference metric'
                ref = imread2tensor(ref, rgb=True)
                ref = ref.unsqueeze(0)
                self.is_valid_input(ref)
        
        self.is_valid_input(target)

        # Efficient device transfer - only if needed
        if target.device != device:
            target = target.to(device, non_blocking=True)
        
        if ref is not None and ref.device != device:
            ref = ref.to(device, non_blocking=True)
            
        return target, ref
    
    def forward(self, target, ref=None, **kwargs):
        # Preprocessing
        target, ref = self._preprocess_input(target, ref)
        
        # Memory optimization context
        if self.memory_efficient and not self.training:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        with torch.set_grad_enabled(self.as_loss):
            # Use mixed precision if enabled and available
            autocast_context = torch.cuda.amp.autocast() if (
                self.enable_mixed_precision and 
                torch.cuda.is_available() and 
                hasattr(torch.cuda.amp, 'autocast')
            ) else torch.nullcontext()
            
            with autocast_context:
                if self.metric_name == 'fid':
                    output = self.net(target, ref, device=self._device_tracker.device, **kwargs)
                elif self.metric_name == 'inception_score':
                    output = self.net(target, device=self._device_tracker.device, **kwargs)
                else:
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        output = self.net(target, ref, **kwargs)
                    elif self.metric_mode == 'NR':
                        output = self.net(target, **kwargs)

        if self.as_loss:
            if isinstance(output, tuple):
                output = output[0]
            return weight_reduce_loss(output, self.loss_weight, self.loss_reduction)
        else:
            return output
