if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    model_name = "NC-AI-consortium-VAETKI/VAETKI"

    temperature, top_p, top_k = 0.7, 0.95, 20
    max_tokens = 32768
    thinking_budget = 16384

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "user", "content": "The sum of two numbers is 20, and their difference is 4. Find the two numbers."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    first_input_ids = tokenizer(prompt, padding=True, return_tensors="pt")["input_ids"].to(llm.device)
    prompt_len = first_input_ids.size(-1)
    first_generation_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=thinking_budget,
    )
    first_output_ids = llm.generate(
        first_input_ids,
        generation_config=first_generation_config,
    )[0][prompt_len:].tolist()
    first_result = tokenizer.decode(first_output_ids, skip_special_tokens=False)

    # Apply thinking budget
    end_token = "<|END|>"
    think_end_token = "</think>"
    early_stopping_text = f"\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n{think_end_token}\n"
    if end_token in first_result:
        final_result = first_result.replace(end_token, "")
    else:
        if think_end_token in first_result:
            thinking_part = first_result

        else:
            thinking_part = first_result + early_stopping_text
        
        second_prompt = prompt + thinking_part
        second_input_ids = tokenizer(second_prompt, padding=True, return_tensors="pt")["input_ids"].to(llm.device)
        second_prompt_len = second_input_ids.size(-1)
        second_generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens - second_prompt_len,
        )
        second_output_ids = llm.generate(
            second_input_ids,
            generation_config=second_generation_config,
        )[0][second_prompt_len:].tolist()
        second_result = tokenizer.decode(second_output_ids, skip_special_tokens=True)

        final_result = thinking_part + second_result
    
    def clean_blocks(text, separators=("\n", "\t")):
        for sep in separators:
            text = sep.join(s[1:] if s.startswith(" ") else s for s in text.split(sep))
        return text

    final_result = clean_blocks(final_result)
    
    thinking_content = final_result.split(think_end_token)[0].strip()
    content = final_result.split(think_end_token)[1].strip()

    print(f"Thinking content: {thinking_content}")
    print(f"Content: {content}")
