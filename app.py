import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import json

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_ID = "hssling/derm-analyzer-adapter"

print("Starting App Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

if ADAPTER_ID:
    print(f"Loading custom fine-tuned LoRA weights: {ADAPTER_ID}")
    try:
        model.load_adapter(ADAPTER_ID)
        print("✅ Adapter loaded successfully over the base Qwen2-VL engine.")
    except Exception as e:
        print(f"Failed to load adapter. Using base model. Error: {e}")

def diagnose_skin(image: Image.Image = None, clinical_notes: str = '', temp: float = 0.4, max_tokens: int = 2000):
    try:
        if image is None:
            return json.dumps({"error": "No image provided."})

        system_prompt = "You are DermaAI, an expert Dermatologist trained extensively on Indian skin types (Fitzpatrick IV-VI) and tropical diseases. Analyze the skin lesion and output a structured clinical report including Findings, Differential Diagnosis, and recommended Indian Pharmacological Management."
        user_prompt = f"Clinical Context: {clinical_notes}\nAnalyze this dermatological image and describe the medical findings, providing treatment and management advice."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=int(max_tokens), temperature=float(temp), top_p=0.9, do_sample=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return output_text

    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=diagnose_skin,
    inputs=[
        gr.Image(type="pil", label="Skin Image"),
        gr.Textbox(label="Clinical Context", value="No additional clinical context provided."),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.4, step=0.1, label="Temperature"),
        gr.Slider(minimum=256, maximum=4096, value=2000, step=256, label="Max Tokens")
    ],
    outputs=gr.Markdown(label="Clinical Report Output"),
    title="DermaAI API (Indian Context)",
    description="Fine-tuned Medical LLM for Dermatology, focused on Fitzpatrick Skin Types IV-VI."
)

if __name__ == "__main__":
    demo.launch()
