from ctransformers import AutoModelForCausalLM

print("Loading...")

llm = AutoModelForCausalLM.from_pretrained(
    ".",
    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama"
)

print("Loaded")

answer = llm(
    "What is acne?",
    max_new_tokens=50
)

print(answer)