if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    model_name = "NC-AI-consortium-VAETKI/VAETKI"
    llm = LLM(
        model=model_name,
        tensor_parallel_size=4,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

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
    first_sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=thinking_budget,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    first_result = llm.generate([first_prompt], first_sampling_params)[0].outputs[0].text

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
        second_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens - thinking_budget,
        )
        second_result = llm.generate([second_prompt], second_sampling_params)[0].outputs[0].text
        final_result = temp_result + second_result
    
    def clean_blocks(text, separators=("\n", "\t")):
        for sep in separators:
            text = sep.join(s[1:] if s.startswith(" ") else s for s in text.split(sep))
        return text

    final_result = clean_blocks(final_result)
    print(f"Result: {final_result}")
