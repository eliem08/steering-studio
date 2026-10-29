import json
import torch
from runtime import load_bundle, OUT
from provenance import build_manifest

def _generate(bundle,prompt,layer=None,dimension=0,scale=0.0,max_new_tokens=40):
 model=bundle.model; batch=bundle.tokenizer(prompt,return_tensors="pt"); batch={k:v.to(bundle.device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
 handle=None
 if layer is not None:
  blocks=getattr(getattr(model,"transformer",None),"h",None)
  if blocks is None: raise ValueError("Steering adapter supports GPT-2-style model.transformer.h blocks only; choose a compatible model.")
  if layer>=len(blocks): raise ValueError("layer index out of range")
  def hook(_m,_i,out):
   if isinstance(out,tuple):
    h=out[0].clone(); h[...,dimension%h.shape[-1]]+=scale; return (h,*out[1:])
   h=out.clone(); h[...,dimension%h.shape[-1]]+=scale; return h
  handle=blocks[layer].register_forward_hook(hook)
 try:
  with torch.no_grad(): ids=model.generate(**batch,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=getattr(bundle.tokenizer,"pad_token_id",None))
  return bundle.tokenizer.decode(ids[0][batch["input_ids"].shape[-1]:],skip_special_tokens=True)
 finally:
  if handle: handle.remove()
def run(text=None,real_model=False,model_id=None):
 b=load_bundle(real_model=real_model,model_id=model_id); prompt=text or "The sky is"
 blocks=getattr(getattr(b.model,"transformer",None),"h",None)
 report={"model":b.model_id,"device":b.device,"prompt":prompt,"baseline":_generate(b,prompt),"interventions":[],"caveat":"Residual-dimension edits are uncalibrated controls, not neuron-specific steering. Measure behavior on held-out tasks and check regressions.","manifest":build_manifest("steering-studio",model_id=b.model_id,device=b.device,config={"max_new_tokens":40,"intervention_scales":[-0.5,0.5,1.0]},inputs={"prompt":prompt})}
 if blocks is not None:
  layer=len(blocks)//2
  for scale in [-0.5,0.5,1.0]: report["interventions"].append({"layer":layer,"dimension":0,"scale":scale,"output":_generate(b,prompt,layer,0,scale)})
  report["interventions"].append({"layer":layer,"dimension":1,"scale":0.5,"control":"alternate-dimension","output":_generate(b,prompt,layer,1,0.5)})
 else: report["warning"]="The selected architecture has no GPT-2-style block adapter; only the baseline ran."
 p=OUT/"steering_report.json"; p.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); report["path"]=str(p); return report
