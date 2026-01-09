if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    model_name = "NC-AI-consortium-VAETKI/VAETKI"
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "user", "content": "Hello, who are you?"},
    ]
    first_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    temperature, top_p, top_k = 0.7, 0.95, 20
    max_tokens = 32768
    thinking_budget = 16384
    first_generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=thinking_budget,
    )

    first_inputs = tokenizer(first_prompt, padding=True, return_tensors="pt")["input_ids"].to(llm.device)
    first_outputs = llm.generate(
        first_inputs,
        generation_config=first_generation_config,
    )
    first_result = tokenizer.batch_decode(
        first_outputs,
        skip_special_tokens=False,
    )[0]

    # Apply thinking budget
    end_token = "<|END|>"
    think_end_token = "</think>"
    early_stopping_text = f"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n{think_end_token}\n"
    if end_token in first_result:
        final_result = first_result.replace(end_token, "")
    else:
        if think_end_token in first_result:
            temp_result = first_result
        else:
            temp_result = f"{first_result}{early_stopping_text}"
        second_prompt = f"{first_prompt}{temp_result}"
        second_generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens - thinking_budget,
        )
        second_outputs = llm.generate(
            second_prompt,
            generation_config=second_generation_config,
        )
        second_result = tokenizer.batch_decode(second_outputs)[0]
        final_result = temp_result + second_result
    
    def clean_blocks(text, separators=("\n", "\t")):
        for sep in separators:
            text = sep.join(s[1:] if s.startswith(" ") else s for s in text.split(sep))
        return text

    final_result = clean_blocks(final_result)
    print(f"Result: {final_result}")
