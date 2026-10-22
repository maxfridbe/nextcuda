"""LTX-2.5's text stage with the official code, on the CPU in bfloat16: the reference nextcuda's port is checked against.

    ref_text.py COMPONENTS OUT_DIR "the prompt" [--negative ""] [--max-len 1024]

What diffusers' LTX2TextEncoderStep and LTX2TextConnectorStep do (modular_pipelines/ltx2/encoders.py): the Gemma tokenizer
(left padding to max_sequence_length), Gemma4UnifiedForConditionalGeneration with output_hidden_states=True, the 49
hidden states stacked, then LTX2TextConnectors on the [negative, prompt] batch. Dumps (numpy):

    ids.npy            int32 [n]               the prompt's real tokens (no padding)
    mask_len.npy       int32 [2]               real tokens of the negative and the prompt
    hs_<i>.npy         float32 [n, 3840]       hidden state i of the prompt's real tokens, i in 0, 1, 12, 24, 47, 48
    hs_all_f16.npy     float16 [n, 49, 3840]   every hidden state of the prompt's real tokens
    video_emb.npy      float32 [1024, 4096]    the connector's video conditioning (prompt)
    audio_emb.npy      float32 [1024, 2048]    the connector's audio conditioning (prompt)
    neg_video_emb.npy, neg_audio_emb.npy       the same for the negative prompt
    conn_mask.npy      int32 [1024]            the connector's binary mask (prompt)
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("components")
    ap.add_argument("out")
    ap.add_argument("prompt")
    ap.add_argument("--negative", default="")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(os.path.join(a.components, "tokenizer"))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(text):
        t = tok([text.strip()], padding="max_length", max_length=a.max_len, truncation=True, add_special_tokens=True, return_tensors="pt")
        return t.input_ids, t.attention_mask

    pid, pmask = encode(a.prompt)
    nid, nmask = encode(a.negative)
    n = int(pmask.sum())
    np.save(os.path.join(a.out, "ids.npy"), pid[0, -n:].numpy().astype(np.int32))
    np.save(os.path.join(a.out, "mask_len.npy"), np.array([int(nmask.sum()), n], np.int32))
    with open(os.path.join(a.out, "prompt.txt"), "w") as f:
        f.write(a.prompt)
    print(f"prompt: {n} tokens; negative: {int(nmask.sum())}", flush=True)

    t0 = time.time()
    te = Gemma4UnifiedForConditionalGeneration.from_pretrained(os.path.join(a.components, "text_encoder"), dtype=torch.bfloat16, low_cpu_mem_usage=True)
    te.eval()
    print(f"text encoder loaded in {time.time() - t0:.0f}s", flush=True)
    outs = []
    with torch.no_grad():
        for ids, mask in ((nid, nmask), (pid, pmask)):
            t0 = time.time()
            o = te(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = torch.stack(o.hidden_states, dim=-1)  # [1, L, 3840, 49]
            print(f"  {len(o.hidden_states)} hidden states in {time.time() - t0:.0f}s", flush=True)
            outs.append((hs.flatten(2, 3).to(torch.bfloat16), mask))
    hsp = outs[1][0].unflatten(2, (3840, 49))[0, -n:]  # [n, 3840, 49]
    np.save(os.path.join(a.out, "hs_all_f16.npy"), hsp.permute(0, 2, 1).float().numpy().astype(np.float16))
    for i in (0, 1, 12, 24, 47, 48):
        np.save(os.path.join(a.out, f"hs_{i}.npy"), hsp[:, :, i].float().numpy())
    del te
    gc.collect()

    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    t0 = time.time()
    conn = LTX2TextConnectors.from_pretrained(os.path.join(a.components, "connectors"), torch_dtype=torch.bfloat16)
    conn.eval()
    print(f"connectors loaded in {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        emb = torch.cat([outs[0][0], outs[1][0]], dim=0)
        mask = torch.cat([outs[0][1], outs[1][1]], dim=0)
        t0 = time.time()
        v, au, m = conn(emb, mask, padding_side="left")
        print(f"connectors in {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(a.out, "neg_video_emb.npy"), v[0].float().numpy())
    np.save(os.path.join(a.out, "neg_audio_emb.npy"), au[0].float().numpy())
    np.save(os.path.join(a.out, "video_emb.npy"), v[1].float().numpy())
    np.save(os.path.join(a.out, "audio_emb.npy"), au[1].float().numpy())
    np.save(os.path.join(a.out, "conn_mask.npy"), m[1].numpy().astype(np.int32))
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
