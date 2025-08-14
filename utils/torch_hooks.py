import torch
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Union, List, Tuple
from collections import defaultdict
from tqdm import tqdm

TensorOrTuple = Union[torch.Tensor, Tuple, Dict, List]

def _get_decoder_layers(model) -> List[torch.nn.Module]:
    """
    Returns a list of decoder blocks irrespective of model family.
    - LLaMA/Mistral/Qwen: model.model.layers
    - GPT-2 style: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError("Could not locate decoder layers on this model.")

def _replace_first_in_output(output: TensorOrTuple, new_hidden: torch.Tensor) -> TensorOrTuple:
    """
    Replace the first 'hidden_states' element in the layer's output with new_hidden.
    Handles Tensor, tuple, list, dict outputs from HF layers.
    """
    if isinstance(output, torch.Tensor):
        return new_hidden
    elif isinstance(output, (tuple, list)):
        # replace element 0
        return type(output)([new_hidden] + list(output)[1:])
    elif isinstance(output, dict):
        # common keys: 'hidden_states' or 'last_hidden_state'
        key = "hidden_states" if "hidden_states" in output else (
              "last_hidden_state" if "last_hidden_state" in output else None)
        if key is None:
            raise ValueError("Dict output without a 'hidden_states'/'last_hidden_state' key.")
        new_out = dict(output)
        new_out[key] = new_hidden
        return new_out
    else:
        raise TypeError(f"Unsupported layer output type: {type(output)}")


class LayerIO:
    """
    Stores per-layer inputs/outputs. You can read these after a forward pass.
    """
    def __init__(self):
        self.pre: Dict[int, torch.Tensor] = {}   # input hidden to layer i
        self.post: Dict[int, torch.Tensor] = {}  # output hidden from layer i

    def clear(self):
        self.pre.clear()
        self.post.clear()


class HookedIntervention:
    """
    Manage forward hooks to:
      - capture residual stream at every decoder layer (pre & post)
      - apply a user-defined intervention on the residual stream of chosen layers
    """
    def __init__(
        self,
        model: torch.nn.Module,
        intervention_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
        layers_to_edit: Optional[List[int]] = None,
        capture_post: bool = True,
        capture_pre: bool = False,
    ):
        """
        Args:
          model: a HF causal LM (AutoModelForCausalLM or similar)
          intervention_fn: function(layer_idx, hidden_states)->modified_hidden_states
                           If None, no edits are applied.
          layers_to_edit: list of layer indices to edit. If None, edit all layers.
          capture_io: if True, stores layer input/output activations in self.io
        """
        self.model = model
        self.layers = _get_decoder_layers(model)
        self.intervention_fn = intervention_fn
        self.layers_to_edit = set(range(len(self.layers))) if layers_to_edit is None else set(layers_to_edit)
        self.capture_post = capture_post
        self.capture_pre = capture_pre
        self.io = LayerIO()
        self._hooks = []
        self._pre_hooks = []

    def _pre_hook(self, idx):
        def hook(module, inputs):
            # 'inputs' is a tuple; for decoder layers the first element is hidden_states
            if not self.capture_pre: 
                return
            if len(inputs) == 0:
                return
            x = inputs[0]
            # store a detached (no-grad) copy to save memory if you don't need backprop
            self.io.pre[idx] = x.detach().cpu()
        return hook

    def _fwd_hook(self, idx):
        def hook(module, inputs, output):
            # Extract the hidden tensor from output
            if isinstance(output, torch.Tensor):
                h = output
            elif isinstance(output, (tuple, list)):
                h = output[0]
            elif isinstance(output, dict):
                h = output.get("hidden_states", output.get("last_hidden_state", None))
                if h is None:
                    raise ValueError("Cannot find hidden_states in dict output.")
            else:
                raise TypeError(f"Unsupported output type at layer {idx}: {type(output)}")

            # Save post if requested
            if self.capture_post:
                self.io.post[idx] = h.detach().cpu()

            # Apply edit if configured
            if self.intervention_fn is not None and idx in self.layers_to_edit:
                h_new = self.intervention_fn(idx, h)
                if h_new is not None and (h_new is not h):
                    return _replace_first_in_output(output, h_new)

            # no change
            return None
        return hook

    @contextmanager
    def activate(self):
        """
        Context manager that installs hooks and ensures removal afterwards.
        """
        try:
            # install hooks
            for i, layer in enumerate(self.layers):
                if self.capture_pre:
                    self._pre_hooks.append(layer.register_forward_pre_hook(self._pre_hook(i), with_kwargs=False))
                self._hooks.append(layer.register_forward_hook(self._fwd_hook(i), with_kwargs=False))
            yield self
        finally:
            # remove hooks
            for h in self._hooks:
                h.remove()
            for h in self._pre_hooks:
                h.remove()
            self._hooks.clear()
            self._pre_hooks.clear()
            
# 3) Define a steering vector and an intervention function
def make_add_vector_intervention(v: torch.Tensor, alpha=-1, t_idx=None):
    """
    Returns an intervention_fn that adds alpha*v to hidden_states[:, t_idx, :].
    v must be on the same device and dtype as the hidden states when applied.
    """
    def intervention(layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D]
        # Broadcast-safe addition
        hidden = hidden.clone()  # avoid in-place on shared tensors
        if t_idx is None:
            hidden = hidden + alpha * v.to(hidden.dtype).to(hidden.device)
        else:
            if t_idx < 0 or t_idx >= hidden.size(1):
                return hidden
            hidden[:, t_idx, :] = hidden[:, t_idx, :] + alpha * v.to(hidden.dtype).to(hidden.device)
        return hidden
    return intervention

def contrast_act_native(model,clean,corrupt,bz =-1 ,avg_before_contrast=True):
    clean_acts = defaultdict(list)
    corrupt_acts = defaultdict(list)
    n = len(clean['input_ids'])
    if bz == -1:
        bz = n
    num_layers = len(model.model.layers)
    edited_layers = list(range(num_layers)) # edit all layers

    clean_hi = HookedIntervention(model, intervention_fn=None, layers_to_edit=edited_layers, capture_post=True)
    corrupt_hi = HookedIntervention(model, intervention_fn=None, layers_to_edit=edited_layers, capture_post=True)
    for i in tqdm(range(0,n,bz),total = n//bz):
        batch_clean = {k: v[i:i+bz] for k,v in clean.items()}
        batch_corrupt = {k: v[i:i+bz] for k,v in corrupt.items()}
        with torch.no_grad():
            with clean_hi.activate():
                _ = model(**batch_clean)
            with corrupt_hi.activate():
                _ = model(**batch_corrupt)
        for l in range(num_layers):
            clean_acts[l].append(clean_hi.io.post[l][:,-1])
            corrupt_acts[l].append(corrupt_hi.io.post[l][:,-1])
        ## empty it out
        clean_hi.io.clear()
        corrupt_hi.io.clear()
        torch.cuda.empty_cache()
       
    clean_acts = {l:torch.cat(v,0).to(model.device) for l,v in clean_acts.items()}
    corrupt_acts = {l:torch.cat(v,0).to(model.device) for l,v in corrupt_acts.items()}
    directions = {}
    for l in range(num_layers):
        if avg_before_contrast:
            directions[l] = corrupt_acts[l].mean(0) - clean_acts[l].mean(0)
        else:
            directions[l] = (corrupt_acts[l]- clean_acts[l]).mean(0)
    return directions


def generate_func(model,prompts,format_fn,gen_kwargs,steer_vec=None,alpha=-1,t_idx=None,layers=[]):
    formatted_prompts = format_fn(prompts)
    if isinstance(layers, int):
        layers = [layers]
    if steer_vec is not None and len(layers):
        # Apply steering vector intervention
        intervention = make_add_vector_intervention(steer_vec, alpha=alpha, t_idx=t_idx)
        hi = HookedIntervention(model, intervention_fn=intervention, layers_to_edit=layers, capture_post=False,capture_pre=False)
        with torch.no_grad():
            with hi.activate():
                out = model.generate(**formatted_prompts, use_cache=True, **gen_kwargs)
        del hi
    else:
        out = model.generate(**formatted_prompts, use_cache=True, **gen_kwargs)
    decoded_tokens = model.tokenizer.batch_decode(out[:,formatted_prompts['input_ids'].shape[1]:], skip_special_tokens=True)
    return decoded_tokens
        