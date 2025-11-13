# llm_reporter.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import os

# -----------------------------
# 1. Tiny Dataset (50 examples)
# -----------------------------
reports = [
    {"input": "MIA: 82%, Clean: 92%, Robust: 68%", "output": "High privacy risk: model leaks membership with 82% attack success. Survives PGD up to ε=0.03. Recommended: differential privacy + adversarial training."},
    {"input": "MIA: 55%, Clean: 90%, Robust: 85%", "output": "Low privacy risk. Model is robust to evasion. Consider monitoring only."},
    {"input": "MIA: 90%, Clean: 88%, Robust: 45%", "output": "Critical privacy breach. Immediate action: add DP noise and retrain with PGD."},
] * 17  # 51 total

dataset = Dataset.from_list(reports)

# -----------------------------
# 2. Tokenizer & Model
# -----------------------------
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# -----------------------------
# 3. Tokenize Function (FIXED)
# -----------------------------
def tokenize_function(examples):
    inputs = [f"Scan: {i}" for i in examples["input"]]
    outputs = [o + tokenizer.eos_token for o in examples["output"]]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=64)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=128)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# -----------------------------
# 4. LoRA Config
# -----------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# -----------------------------
# 5. Training Args (FIXED BATCH SIZE)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./report_model",
    per_device_train_batch_size=2,  # Fixed: was 4 → caused mismatch
    num_train_epochs=3,
    save_steps=50,
    logging_steps=10,
    learning_rate=5e-4,
    fp16=False,
    report_to=[],
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -----------------------------
# 6. Train & Save (Only Once)
# -----------------------------
if not os.path.exists("./report_model"):
    print("Training LoRA model for reports...")
    trainer.train()
    model.save_pretrained("./report_model")
    tokenizer.save_pretrained("./report_model")
    print("Report model saved!")
else:
    print("Report model already exists. Loading...")

# -----------------------------
# 7. Generate Report Function
# -----------------------------
def generate_report(results):
    model = AutoModelForCausalLM.from_pretrained("./report_model")
    tokenizer = AutoTokenizer.from_pretrained("./report_model")
    model.eval()

    prompt = f"Scan: MIA: {results['mia_accuracy']}%, Clean: {results['clean_accuracy']}%, Robust: {results['robust_accuracy']}%"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = text.split("Scan:")[1].split("Recommended")[0].strip() if "Recommended" in text else text

    # PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AI Model Privacy & Robustness Report</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Membership Leak:</b> {results['mia_accuracy']}% attack success", styles['Normal']))
    story.append(Paragraph(f"<b>Clean Accuracy:</b> {results['clean_accuracy']}%", styles['Normal']))
    story.append(Paragraph(f"<b>Robust Accuracy (PGD ε=0.03):</b> {results['robust_accuracy']}%", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>AI Summary:</b>", styles['Normal']))
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Recommended: differential privacy + adversarial training.", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer