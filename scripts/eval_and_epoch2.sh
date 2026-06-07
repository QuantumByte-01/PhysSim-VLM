#!/bin/bash
# Wait for epoch 1 → evaluate on physics val → start epoch 2

cd /workspace/PhysSim-VLM
export HF_TOKEN=YOUR_HF_TOKEN_HERE
export WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
export WANDB_PROJECT=PhysSim-VLM
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false

TRAIN_PID=20596

echo "============================================"
echo " Post-Epoch 1 Pipeline"
echo " $(date)"
echo "============================================"

# ── 1. Wait for epoch 1 to finish ────────────────────────────────
echo "[1/3] Waiting for epoch 1 training (PID $TRAIN_PID)..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
done
echo " Epoch 1 training done at $(date)"
echo ""

# Verify checkpoint
CKPT="checkpoints/lora_sft_epoch1/final"
if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
    echo "ERROR: No checkpoint found at $CKPT"
    exit 1
fi

# ── 2. Evaluate on physics val set ───────────────────────────────
echo "[2/3] Running physics val evaluation..."
python3 - <<'PYEOF'
import os, json, re, base64, io, random, torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

HF_TOKEN = os.environ["HF_TOKEN"]
ROOT = Path("/workspace/PhysSim-VLM")
CKPT = ROOT / "checkpoints" / "lora_sft_epoch1" / "final"
RESULTS = ROOT / "results" / "sft_epoch1"
RESULTS.mkdir(parents=True, exist_ok=True)
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MAX_EVAL = 300 # subset for speed

print("Loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)

print("Loading base model + LoRA weights...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, token=HF_TOKEN, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model = PeftModel.from_pretrained(model, str(CKPT))
model.eval()

# Remove grouped_mm on ROCm
if torch.cuda.is_available() and hasattr(torch.version, "hip"):
    if hasattr(torch, "_grouped_mm"):
        del torch._grouped_mm
    if hasattr(torch.nn.functional, "grouped_mm"):
        del torch.nn.functional.grouped_mm

print("Loading val dataset from HuggingFace...")
ds = load_dataset("Swastikr/PhysSim-VLM-Dataset", token=HF_TOKEN, split="val")
indices = list(range(len(ds)))
random.seed(42)
random.shuffle(indices)
ds = ds.select(indices[:MAX_EVAL])
print(f"Evaluating {len(ds)} val samples...")

def decode_frames(frames_b64):
    imgs = []
    for b64 in frames_b64:
        imgs.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    return imgs

def score_sample(task, pred, answer):
    pred = pred.strip().lower()
    # Extract answer tag if present
    m = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
    if m:
        pred = m.group(1).strip()
    if task == "stability":
        return 1.0 if pred == answer.strip().lower() else 0.0
    elif task == "ttc":
        try:
            p, g = float(pred.split()[0]), float(answer)
            return float(torch.exp(torch.tensor(-3 * abs(p - g) / max(g, 0.01))).item())
        except:
            return 0.0
    elif task == "trajectory":
        try:
            px = float(re.search(r"x=([-\d.]+)", pred).group(1))
            py = float(re.search(r"y=([-\d.]+)", pred).group(1))
            gx = float(re.search(r"x=([-\d.]+)", answer).group(1))
            gy = float(re.search(r"y=([-\d.]+)", answer).group(1))
            dist = ((px-gx)**2 + (py-gy)**2)**0.5
            return float(torch.exp(torch.tensor(-dist / 2.0)).item())
        except:
            return 0.0
    return 0.0

predictions = []
task_scores = defaultdict(list)

for i, sample in enumerate(ds):
    task = sample["task"]
    prompt = sample["prompt"]
    answer = sample["answer"]
    frames = decode_frames(sample["frames_b64"])

    user_content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=text, images=frames,
        return_tensors="pt", max_length=2048,
        truncation=True, max_pixels=401408,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    pred = processor.decode(new_tokens, skip_special_tokens=True)
    score = score_sample(task, pred, answer)
    task_scores[task].append(score)

    predictions.append({
        "scene_id": sample["scene_id"],
        "task": task,
        "difficulty": sample["difficulty"],
        "answer": answer,
        "prediction": pred,
        "score": score,
    })

    if (i + 1) % 20 == 0:
        print(f" {i+1}/{len(ds)} done...")

# Save predictions
with open(RESULTS / "predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

# Compute and print results
print("\n" + "="*50)
print(" Physics Val Results - Epoch 1 SFT")
print("="*50)
all_scores = []
md_lines = ["# PhysSim-VLM - Epoch 1 SFT Results\n",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"Checkpoint: {CKPT}\n\n",
            "| Task | Samples | Score |\n", "|------|---------|-------|\n"]

for task, scores in task_scores.items():
    avg = sum(scores) / len(scores)
    all_scores.extend(scores)
    print(f" {task:12s}: {avg:.4f} ({len(scores)} samples)")
    md_lines.append(f"| {task} | {len(scores)} | {avg:.4f} |\n")

overall = sum(all_scores) / len(all_scores)
print(f" {'overall':12s}: {overall:.4f} ({len(all_scores)} samples)")
md_lines.append(f"| **overall** | **{len(all_scores)}** | **{overall:.4f}** |\n")
print("="*50)

with open(RESULTS / "results.md", "w") as f:
    f.writelines(md_lines)

print(f"\nResults saved to {RESULTS}/")

# ── Write detailed results.md ─────────────────────────────────────
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = [
    "# PhysSim-VLM - SFT Epoch 1 Results\n\n",
    f"**Model:** Qwen/Qwen3-VL-30B-A3B-Instruct + LoRA SFT (rank=128)\n",
    f"**Date:** {now}\n",
    f"**Checkpoint:** checkpoints/lora_sft_epoch1/final\n",
    f"**Dataset:** Swastikr/PhysSim-VLM-Dataset (15k scenes: TTC + Trajectory + Stability)\n",
    f"**Samples evaluated:** {len(predictions)}\n\n",
    "## Overall Score\n\n",
    f"**Score: {overall:.4f}** ({overall*100:.1f}%)\n\n",
    "## Per-Task Breakdown\n\n",
    "| Task | Samples | Score | Metric |\n",
    "|------|---------|-------|--------|\n",
]
metric_desc = {"ttc": "exp(-3|pred-gt|/gt)", "stability": "exact match", "trajectory": "exp(-dist/2)"}
for task, scores in sorted(task_scores.items()):
    avg = sum(scores)/len(scores)
    lines.append(f"| {task} | {len(scores)} | {avg:.4f} ({avg*100:.1f}%) | {metric_desc.get(task,'')} |\n")
lines.append(f"| **overall** | **{len(all_scores)}** | **{overall:.4f} ({overall*100:.1f}%)** | weighted avg |\n\n")

lines += [
    "## Comparison vs Baseline\n\n",
    "| | Baseline (zero-shot) | SFT Epoch 1 | Delta |\n",
    "|--|---------------------|-------------|-------|\n",
    "| Dynamics (PhysBench) | 18.3% | - (eval pending) | - |\n",
    "| Overall (PhysBench) | 40.7% | - (eval pending) | - |\n\n",
    "## Training Config\n\n",
    "| Param | Value |\n", "|-------|-------|\n",
    "| LoRA rank | 128 |\n",
    "| LoRA alpha | 256 |\n",
    "| Batch size | 1 (grad_accum=16, effective=16) |\n",
    "| Learning rate | 2e-4 (cosine) |\n",
    "| Epochs | 1 |\n",
    "| Training samples | 12,023 |\n",
    "| GPU | AMD MI300x (205 GB HBM3) |\n",
]

with open(RESULTS / "results.md", "w") as f:
    f.writelines(lines)
print(f"Results MD saved to {RESULTS}/results.md")
PYEOF

echo ""
echo "[3/4] Running PhysBench evaluation (zero-shot + fine-tuned)..."
# Wait for image download to finish if still running
while kill -0 52506 2>/dev/null; do
    echo " Waiting for PhysBench images to finish downloading..."
    sleep 30
done

IMG_COUNT=$(ls /workspace/PhysSim-VLM/physbench/image/ 2>/dev/null | wc -l)
echo " PhysBench images available: $IMG_COUNT"

if [ "$IMG_COUNT" -gt "100" ]; then
    python3 scripts/eval_physbench.py \
        --use_existing_baseline \
        --limit 500 \
        > /tmp/physbench_eval.log 2>&1
    echo " PhysBench eval complete. See /tmp/physbench_eval.log"
else
    echo " WARNING: PhysBench images not available, skipping PhysBench eval."
fi

echo ""
echo "[4/4] Uploading LoRA weights to HuggingFace (private)..."
python3 - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
CKPT = Path("/workspace/PhysSim-VLM/checkpoints/lora_sft_epoch1/final")
REPO_ID = "Swastikr/PhysSim-VLM-Qwen3VL-30B-LoRA"

api = HfApi(token=HF_TOKEN)

# Ensure repo exists and is private
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
    print(f"Repo ready: {REPO_ID} (private)")
except Exception as e:
    print(f"Repo check: {e}")

# Upload all checkpoint files
print(f"Uploading from {CKPT} ...")
api.upload_folder(
    folder_path=str(CKPT),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add PhysSim-VLM LoRA SFT epoch 1 weights (rank=128, 15k physics scenes)",
)
print(f"Done! Weights live at: https://huggingface.co/{REPO_ID}")
PYEOF

echo ""
echo "[4/4] Pushing results to GitHub..."
git -C /workspace/PhysSim-VLM add results/sft_epoch1/ results/physbench/ scripts/eval_physbench.py
git -C /workspace/PhysSim-VLM commit -m "$(cat <<'EOF'
results: SFT epoch 1 eval - physics val + PhysBench comparison

- results/sft_epoch1/results.md: per-task scores (TTC, Stability, Trajectory)
- results/sft_epoch1/predictions.json: 300 val sample predictions
- results/physbench/zero_shot/results.md: base model PhysBench scores
- results/physbench/sft_epoch1/results.md: fine-tuned PhysBench scores
- results/physbench/comparison.md: side-by-side delta table
- scripts/eval_physbench.py: PhysBench eval script (zero-shot + LoRA)

Model: Qwen3-VL-30B + LoRA rank=128, trained on 15k physics scenes

EOF
)"
git -C /workspace/PhysSim-VLM push https://YOUR_GITHUB_PAT_HERE@github.com/QuantumByte-01/PhysSim-VLM.git master
echo "Pushed to GitHub!"

echo ""
echo "============================================"
echo " All done at $(date)"
echo " Eval results : results/sft_epoch1/results.md"
echo " HF model : Swastikr/PhysSim-VLM-Qwen3VL-30B-LoRA (private)"
echo " GitHub : QuantumByte-01/PhysSim-VLM"
echo "============================================"
