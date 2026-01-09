# Download the model
Download [VAETKI](https://huggingface.co/NC-AI-consortium-VAETKI/VAETKI) on Hugging Face

# Quickstart

> **Hardware requirement**  
> VAETKI inference requires **at least 4× NVIDIA A100 GPUs**.

## 🤗 Transformers
- **Required version**: `transformers>=4.56`
- **Recommended version**: `transformers==4.57.3`
### Install dependencies
```bash
pip install "transformers>=4.56"
pip install flash-attn --no-build-isolation
```
### Example code
Please refer to the [Transformers example](https://github.com/wbl-ncai/VAETKI/blob/releases/v1.0.0/example.py)

## vLLM
- **Supported version**: `vllm==0.11.2`

### Install dependencies
```bash
pip install "vllm==0.11.2"
pip install "git+https://github.com/wbl-ncai/VAETKI.git@releases/v1.0.0#subdirectory=vllm_plugin"
```
For convenience, you may use the following vLLM Docker image:
```bash
docker pull vllm/vllm-openai:v0.11.2
```

### Example code
Please refer to the [vLLM example](https://github.com/wbl-ncai/VAETKI/blob/releases/v1.0.0/vllm_plugin/vaetki/__main__.py)

# License
This code repository is licensed under the MIT License. The use of VAETKI models is subject to the Model License.

# Contact
If you are interested to leave a message or have any questions, please contact us at wbl.ncai.hf@gmail.com.
