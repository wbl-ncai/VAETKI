def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "VaetkiForCausalLM",
        "vaetki.model:VaetkiForCausalLM",
    )
