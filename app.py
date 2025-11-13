# app.py
import gradio as gr
from scanner import run_full_scan
from llm_reporter import generate_report
import torch

def scan_model(model_file):
    torch.save(models.resnet18(pretrained=True).state_dict(), "temp.pth")
    results, _ = run_full_scan("temp.pth")
    pdf = generate_report(results)
    return results, pdf

with gr.Blocks() as demo:
    gr.Markdown("# AI Model Privacy Scanner")
    gr.Markdown("Upload a `.pth` ResNet18-CIFAR10 model → Get privacy & robustness report in <60s")

    with gr.Row():
        model_input = gr.File(label="Upload .pth model")
        submit = gr.Button("Scan Model")

    with gr.Row():
        output_text = gr.JSON(label="Results")
        output_pdf = gr.File(label="Download Report")

    submit.click(
        fn=scan_model,
        inputs=model_input,
        outputs=[output_text, output_pdf]
    )

demo.launch()