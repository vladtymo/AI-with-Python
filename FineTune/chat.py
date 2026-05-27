import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./finetuned-tinyllama-1.1b-chat"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
)

model.eval()

print("AI is ready. Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt from output
    answer = response.split("### Response:\n")[-1]

    print(f"\nAI: {answer}\n")