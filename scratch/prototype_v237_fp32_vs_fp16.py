import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

ROBUST_TEXT = """
The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. The four inner system planets—Mercury, Venus, Earth and Mars—are terrestrial planets, being primarily composed of rock and metal.

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term AI is also used to describe a property of machines which mimic cognitive functions that humans associate with the human mind.

The Industrial Revolution was a period of global economic transition towards more efficient and stable manufacturing processes that succeeded the Agricultural Revolution. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools and the rise of the mechanized factory system.
"""

def evaluate_robust_ppl(model, tokenizer, text, device):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss)
    return loss.item(), ppl.item()

def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"\n--- Experimento v237: FP32 vs FP16 Precision Check ---")
    
    # 1. EVALUACIÓN FP32
    print("Cargando modelo en Float32...", end=" ", flush=True)
    model_fp32 = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    loss32, ppl32 = evaluate_robust_ppl(model_fp32, tokenizer, ROBUST_TEXT, device)
    print(f"PPL: {ppl32:.6f}")
    
    # 2. EVALUACIÓN FP16
    print("Cargando modelo en Float16...", end=" ", flush=True)
    model_fp16 = AutoModelForCausalLM.from_pretrained(model_name).to(device).half()
    loss16, ppl16 = evaluate_robust_ppl(model_fp16, tokenizer, ROBUST_TEXT, device)
    print(f"PPL: {ppl16:.6f}")
    
    # 3. COMPARATIVA
    print("\n--- RESULTADOS ---")
    print(f"Formato | Perplexity | Tamaño Relativo")
    print(f"FP32    | {ppl32:<10.6f} | 100% (324 MB)")
    print(f"FP16    | {ppl16:<10.6f} |  50% (162 MB)")
    
    delta = ppl16 - ppl32
    print(f"\nDelta PPL (FP16 - FP32): {delta:+.6f}")
    
    if abs(delta) < 0.01:
        print("✅ CONCLUSIÓN: El paso a FP16 es prácticamente gratuito en términos de inteligencia.")
    else:
        print("⚠️ AVISO: Hay una ligera degradación detectable en FP16.")

if __name__ == "__main__":
    main()
