import torch

def generate(tokenizer, model, prompt, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **kwargs
        ).to('cpu')

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
