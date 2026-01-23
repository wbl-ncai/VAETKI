# Model Release

**VAETKI model** is available on **[Hugging Face](https://huggingface.co/NC-AI-consortium-VAETKI/VAETKI)**.  
See the **[Technical Report](https://github.com/wbl-ncai/VAETKI/blob/releases/v1.0.0/VAETKI_Technical_Report.pdf)** for details.


# Quickstart
> **Hardware requirement**  
> VAETKI inference requires **at least 4 GPUs with 80GB VRAM**
> (e.g., **NVIDIA A100 80GB ×4**).

## 🤗 Transformers
VAETKI can be run with Hugging Face Transformers.
- **Required version**: `transformers>=4.56`
- **Recommended version**: `transformers~=4.57.3`

### Install dependencies
```bash
pip install transformers>=4.56 accelerate>=1.10
pip install flash-attn>=2.8 --no-build-isolation
```

### Example code
Please refer to the [Transformers example code](https://github.com/wbl-ncai/VAETKI/blob/releases/v1.0.0/example.py).


## vLLM
VAETKI also supports inference via vLLM.
- Tested with: `vllm==0.11.2`

For convenience, you may use the following vLLM Docker image:
``` bash
docker pull vllm/vllm-openai:v0.11.2
```

### Install VAETKI vLLM plugin
``` bash
pip install "git+https://github.com/wbl-ncai/VAETKI.git@releases/v1.0.0#subdirectory=vllm_plugin"
```

### Example code
Please refer to the [vLLM example code](https://github.com/wbl-ncai/VAETKI/blob/releases/v1.0.0/vllm_plugin/vaetki/__main__.py).

# Citation

If you use VAETKI or refer to our work in your research, please cite the following technical report:

```bibtex
@misc{ncai2025vaetkitechnicalreport,
  title        = {VAETKI Technical Report},
  author       = {{NC-AI Consortium}},
  year         = {2025},
  howpublished = {\url{https://github.com/wbl-ncai/VAETKI/blob/releases/v1.0.0/VAETKI_Technical_Report.pdf}},
  note         = {Version 1.0.0}
}
```

# License
This code repository is licensed under the MIT License. The use of VAETKI models is subject to the Model License.

# Contact
If you are interested to leave a message or have any questions, please contact us at wbl.ncai.hf@gmail.com.
