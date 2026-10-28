"""Experimental activation steering for GPT-2-style transformer blocks.

Computes a direction from positive vs. contrast prompts and adds a scaled direction
to one selected block output during generation. Use tiny scales and compare controls.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from provenance import build_manifest

def device_name(requested):
    if requested!="auto": return requested
    return "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"

def run(model_id="gpt2", positive="I feel confident, clear, and decisive.", contrast="I feel uncertain, confused, and hesitant.", prompt="I feel", layer=6, scale=0.5, output="outputs/steering", device="auto", max_new_tokens=60):
    device=device_name(device);tok=AutoTokenizer.from_pretrained(model_id);model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto").to(device).eval()
    blocks=getattr(getattr(model,"transformer",None),"h",None)
    if blocks is None: raise ValueError("This first implementation supports GPT-2-style models with model.transformer.h blocks.")
    if layer<0 or layer>=len(blocks): raise ValueError(f"layer must be between 0 and {len(blocks)-1}")
    captured={}
    def capture(name):
        def hook(module, inputs, output):
            value=output[0] if isinstance(output,tuple) else output
            captured[name]=value.detach().float().mean(dim=1).squeeze(0)
        return hook
    handle=blocks[layer].register_forward_hook(capture("v"))
    def mean_activation(text):
        captured.clear();x=tok(text,return_tensors="pt").to(device)
        with torch.inference_mode(): model(**x,use_cache=False)
        if "v" not in captured: raise RuntimeError("Activation hook did not capture a tensor")
        return captured["v"]
    try: direction=mean_activation(positive)-mean_activation(contrast)
    finally: handle.remove()
    norm=direction.norm().clamp_min(1e-8);direction=direction/norm
    def generate(injection_scale):
        def steering_hook(module, inputs, output):
            value=output[0] if isinstance(output,tuple) else output
            delta=(direction.to(device=value.device,dtype=value.dtype)*injection_scale).view(1,1,-1)
            changed=value+delta
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        h=blocks[layer].register_forward_hook(steering_hook)
        try:
            x=tok(prompt,return_tensors="pt").to(device)
            with torch.inference_mode():
                y=model.generate(**x,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
            return tok.decode(y[0][x["input_ids"].shape[1]:],skip_special_tokens=True)
        finally:h.remove()
    baseline=generate(0.0);steered=generate(scale)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report={"model_id":model_id,"device":device,"layer":layer,"scale":scale,"direction_norm_before_normalization":float(norm.cpu()),"positive_prompt":positive,"contrast_prompt":contrast,"generation_prompt":prompt,"baseline":baseline,"steered":steered,"manifest":build_manifest("steering-studio",model_id=model_id,device=device,config={"layer":layer,"scale":scale,"max_new_tokens":max_new_tokens},inputs={"positive":positive,"contrast":contrast,"prompt":prompt}),"warning":"Exploratory intervention only; semantic direction and effects are not validated. Steering can degrade unrelated capabilities."}
    (out/"steering.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    return report
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default="gpt2");p.add_argument("--positive",default="I feel confident, clear, and decisive.");p.add_argument("--contrast",default="I feel uncertain, confused, and hesitant.");p.add_argument("--prompt",default="I feel");p.add_argument("--layer",type=int,default=6);p.add_argument("--scale",type=float,default=0.5);p.add_argument("--output",default="outputs/steering");p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto");p.add_argument("--max-new-tokens",type=int,default=60)
    a=p.parse_args();r=run(a.model_id,a.positive,a.contrast,a.prompt,a.layer,a.scale,a.output,a.device,a.max_new_tokens);print(json.dumps({"baseline":r["baseline"],"steered":r["steered"],"output":a.output},indent=2))
if __name__=="__main__":main()
