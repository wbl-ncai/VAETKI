if __name__ == "__main__":
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    from qwen_vl_utils import process_vision_info
    import os

    model_path = "nc-ai-consortium/VAETKI-VL-7B-A1B"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    test_image = os.environ.get("TEST_IMAGE", "https://huggingface.co/nc-ai-consortium/VAETKI-VL-7B-A1B/resolve/main/demo.jpg")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": test_image,
                },
                {"type": "text", "text": "What color is the lighting on the main support structure of the Ferris wheel?"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # Model Output Example: 'The lighting on the main support structure of the Ferris wheel is red.'
    print(output_text)