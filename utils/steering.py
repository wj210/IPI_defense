from utils.utils import *
import torch
from collections import defaultdict

def contrast_activations(model,clean,corrupt,avg_before_contrast=True):
    clean_acts = {}
    corrupt_acts = {}
    with torch.no_grad(), model.trace(clean) as tracer:
        for l in range(len(model.model.layers)):
            clean_acts[l] = model.model.layers[l].output[0][:,-1].save()
    clear_mem()
    with torch.no_grad(), model.trace(corrupt) as tracer:
        for l in range(len(model.model.layers)):
            corrupt_acts[l] = model.model.layers[l].output[0][:,-1].save()
    clear_mem()
    directions = {}
    for l in range(len(model.model.layers)):
        if avg_before_contrast:
            directions[l] = corrupt_acts[l].mean(0) - clean_acts[l].mean(0)
        else:
            directions[l] = (corrupt_acts[l]- clean_acts[l]).mean(0)
    return directions

def get_steering_vec(model,corrupt,clean,bz=-1,return_separate_vectors = False): # if return_separate_vectors, return corrupt_vec and clean vec as well.
    if bz == -1:
        bz = len(corrupt)
    all_corrupt_acts,all_clean_acts = defaultdict(list),defaultdict(list)
    for i in range(0,len(corrupt),bz):
        corrupt_batch = corrupt[i:i+bz]
        clean_batch = clean[i:i+bz]
        corrupt_inps = encode_fn(model,corrupt_batch,return_attn=True)
        clean_inps = encode_fn(model,clean_batch,return_attn=True)

        _,corrupt_cache = model.run_with_cache(corrupt_inps.input_ids,attention_mask = corrupt_inps.attention_mask,names_filter = resid_name_filter)
        _,clean_cache = model.run_with_cache(clean_inps.input_ids,attention_mask = clean_inps.attention_mask,names_filter = resid_name_filter)

        for k in corrupt_cache.keys():
            all_corrupt_acts[retrieve_layer_fn(k)].append(corrupt_cache[k][:,-1])
            all_clean_acts[retrieve_layer_fn(k)].append(clean_cache[k][:,-1])
        del corrupt_cache,clean_cache
        torch.cuda.empty_cache()
    all_corrupt_acts = {k:torch.concat(v,dim = 0) for k,v in all_corrupt_acts.items()}
    all_clean_acts = {k:torch.concat(v,dim = 0) for k,v in all_clean_acts.items()}
    steering_vec = {k: all_corrupt_acts[k].mean(0)- all_clean_acts[k].mean(0) for k in all_corrupt_acts.keys()}

    if not return_separate_vectors:
        return steering_vec
    else:
        return steering_vec, (all_corrupt_acts,all_clean_acts)