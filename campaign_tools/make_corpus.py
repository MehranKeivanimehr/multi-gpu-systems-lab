import hashlib, json, random, sys
REV = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT_LEN, N_PROMPTS, SEED = 256, 64, 20261016
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
vocab = int(tok.vocab_size); special = {i for i in tok.all_special_ids}
rng = random.Random(SEED); lo, hi = 10, vocab - 100
prompts = []
for _ in range(N_PROMPTS):
    ids = []
    while len(ids) < PROMPT_LEN:
        t = rng.randint(lo, hi)
        if t not in special: ids.append(t)
    prompts.append(ids)
assert all(0 <= t < vocab for p in prompts for t in p)
_ = tok.decode(prompts[0][:16])
payload = {
 "schema":"titan-prompt-corpus/1","model":MODEL,"revision":REV,"tokenizer_revision":REV,
 "vocab_size":vocab,"seed":SEED,"prompt_len":PROMPT_LEN,"n_prompts":N_PROMPTS,
 "excluded_special_ids":sorted(special),
 "policy":{"greedy":True,"temperature":0.0,"ignore_eos":True,
           "exact_output_tokens":128,"streaming_output_tokens":64,
           "enable_prefix_caching":False},
 "arm_mapping":{"hf_batches":[1,4,16],"vllm_request_batches":[1,4,16,64],
   "note":"HF 1/4/16 match vLLM 1/4/16 request batches on the SAME prompts; "
          "vLLM 64 is an ADDITIONAL arm with no HF-64 baseline and must not be "
          "compared across engines"},
 "prompts":prompts}
blob = json.dumps(payload, sort_keys=True, separators=(",",":")).encode()
payload["corpus_sha256"] = hashlib.sha256(blob).hexdigest()
with open(sys.argv[1],"w") as fh: json.dump(payload, fh, indent=1, sort_keys=True)
ph = hashlib.sha256(json.dumps(prompts, separators=(",",":")).encode()).hexdigest()
print(f"  vocab_size        : {vocab}")
print(f"  prompts           : {N_PROMPTS} x {PROMPT_LEN} tokens")
print(f"  prompt_ids_sha256 : {ph}")
print(f"  corpus_sha256     : {payload['corpus_sha256']}")
print(f"  file_sha256       : {hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()}")
