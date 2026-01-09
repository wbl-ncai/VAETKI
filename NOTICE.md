# Third Party Notices

This repository makes use of third-party open-source software and data.
The following is a list of the components used and their respective licenses.

---

## 🛠 Software Libraries

### 1. Hugging Face Transformers
A library for state-of-the-art Natural Language Processing.

- **Website:** [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- **License:** Apache License 2.0
- **Copyright:** © 2018- Hugging Face Inc.
> Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

### 2. vllm 
a fast and easy-to-use library for LLM inference and serving

- **Website:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **License:** Apache License 2.0
> Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

### 3. NVIDIA Megatron-LM
GPU-optimized library for training transformer models at scale

- **Website:** [https://github.com/NVIDIA/Megatron-LM/tree/main](https://github.com/NVIDIA/Megatron-LM/tree/main)
- **License:** Apache License 2.0
- **Copyright:** Copyright (c) 2019-2024 NVIDIA CORPORATION. All rights reserved.
> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

---
---

## 📊 Datasets

### 1. NVIDIA Datasets
Datasets used for internal model training under the NVIDIA Data Agreement.
**Note:** These datasets are not distributed with this software.

- **Licensor:** NVIDIA Corporation
- **License:** NVIDIA Data Agreement for Model Training (v. August 15, 2025)
- **Copyright:** Copyright © 2025 NVIDIA Corporation. All rights reserved.

> [cite_start]**WARRANTY DISCLAIMER** 
>
> THE DATASETS ARE PROVIDED “AS IS”. TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, NVIDIA DISCLAIMS ALL WARRANTIES AND REPRESENTATIONS OF ANY KIND, WHETHER EXPRESS, IMPLIED OR STATUTORY, RELATING TO OR ARISING UNDER THIS AGREEMENT, INCLUDING, WITHOUT LIMITATION, THE WARRANTIES OF TITLE, NONINFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, USAGE OF TRADE AND COURSE OF DEALING.

### 2. EleutherAI Proof-Pile-2
A dataset of mathematical and scientific documents (arXiv, OpenWebMath, AlgebraicStack).

- **Source:** https://huggingface.co/datasets/EleutherAI/proof-pile-2
- **License:** Various (Composite) - adhering to source data licenses.
- **Copyright:**
  - OpenWebMath: ODC-By 1.0
  - RedPajama (arXiv): Various (per-paper license)
  - AlgebraicStack: Various (per-repository license)

> **Notice:** This dataset is a compilation of multiple sources. The original licenses of the underlying data apply.
>
> **Attribution for OpenWebMath Component:**
> This project contains information from "OpenWebMath" which is made available under the ODC Attribution License.
>
> **Citations:**
> * Azerbayev, et al., "Llemma: An Open Language Model For Mathematics", arXiv:2310.10631, 2023.
> * Paster, et al., "OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text", arXiv:2310.06786, 2023.

### 3. The Stack v2 (Smol IDs)
A dataset containing identifiers for permissive code repositories from Software Heritage.

- **Source:** https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids
- **License:** ODC-By 1.0 (Dataset Compilation) & Various Permissive Licenses (Underlying Code)
- **Copyright:** © 2024 BigCode Project & Software Heritage

> **Dataset License:**
> The dataset aggregation is licensed under the Open Data Commons Attribution License v1.0 (ODC-By 1.0).
>
> **Content License:**
> The code referenced in this dataset is subject to various permissive licenses (e.g., MIT, Apache 2.0, BSD) as indicated in the original repositories.
>
> **Attribution & Citation:**
> This project uses "The Stack v2" dataset.
> * Lozhkov, A., et al. "StarCoder 2 and The Stack v2: The Next Generation", arXiv:2402.19173, 2024.

### 4. Stack-Edu
A filtered subset of The Stack v2 containing high-quality educational code.

- **Source:** https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- **License:** ODC-By 1.0 (Dataset Compilation) & Various Permissive Licenses (Underlying Code)
- **Copyright:** © 2025 Hugging Face (HuggingFaceTB)

> **Dataset License:**
> The dataset aggregation is licensed under the Open Data Commons Attribution License v1.0 (ODC-By 1.0).
>
> **Content License:**
> The code referenced in this dataset is filtered from "The Stack v2" and is subject to various permissive licenses (e.g., MIT, Apache 2.0, BSD) as indicated in the original repositories.
>
> **Attribution & Citation:**
> This project uses the "Stack-Edu" dataset.
> * Allal, L. B., et al. "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model", arXiv:2502.02737, 2025.
> * Lozhkov, A., et al. "StarCoder 2 and The Stack v2: The Next Generation", arXiv:2402.19173, 2024.

### 5. DCLM-baseline-1.0
A 4T token / 3B document pretraining dataset that achieves strong performance on language model benchmarks.

- **Source:** https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Copyright:** © ML Foundations

> Licensed under the Creative Commons Attribution 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
>
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 6. WanJuan-Korean
A 4T token / 3B document pretraining dataset that achieves strong performance on language model benchmarks.

- **Source:** https://huggingface.co/datasets/opendatalab/WanJuan-Korean
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Copyright:** © OpenDataLab

> Licensed under the Creative Commons Attribution 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
>
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 7. AceReason-1.1-SFT
A diverse and high-quality supervised fine-tuning dataset focused on math and code reasoning.

- **Source:** https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Copyright:** Copyright © 2025 NVIDIA Corporation. All rights reserved.

> Licensed under the Creative Commons Attribution 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
>
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 8. OpenScience
A multi-domain synthetic dataset designed to improve general-purpose reasoning in large language models.

- **Source:** https://huggingface.co/datasets/nvidia/OpenScience
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Copyright:** Copyright © 2025 NVIDIA Corporation. All rights reserved.

> Licensed under the Creative Commons Attribution 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
>
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 9. Nemotron-PrismMath
A state-of-the-art math reasoning dataset with diverse, novel math problems.

- **Source:** https://huggingface.co/datasets/nvidia/Nemotron-PrismMath
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Copyright:** Copyright © 2025 NVIDIA Corporation. All rights reserved.

> Licensed under the Creative Commons Attribution 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
>
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 10. OpenCodeGeneticInstruct
A dataset comprising more than 15M coding instructions in python which is generated synthetically with the Genetic-Instruct approach.

- **Source:** https://huggingface.co/datasets/nvidia/OpenCodeGeneticInstruct
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Copyright:** © Copyright © 2025 NVIDIA Corporation. All rights reserved.

> Licensed under the Creative Commons Attribution 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
>
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 11. StackExchange_Mar2023
A dataset containing StackExchange questions and answers.

- **Source:** https://huggingface.co/datasets/HuggingFaceGECLM/StackExchange_Mar2023
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Copyright:** © HuggingFaceGECLM
> Licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
> To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
>
>**License Scope Note:**
> The CC BY-SA 4.0 license applies to the dataset itself. The model weights provided in this repository are generated through statistical analysis and pattern recognition of the dataset and are distributed under the [MIT/Apache] license as a separate work, consistent with applicable copyright exceptions for text and data mining.
> **Disclaimer:** THIS MATERIAL IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### 12. MegaMath
An open math pretraining dataset curated from diverse, math-focused sources, with over 300B tokens.

- **Source:** https://huggingface.co/datasets/LLM360/MegaMath
- **License:** Open Data Commons Attribution License v1.0 (ODC-By 1.0)
- **Copyright:** @ LLM360
> Licensed under the Open Data Commons Attribution License v1.0.
> To view a copy of this license, visit https://opendatacommons.org/licenses/by/1.0/
>
> **Attribution:** This project contains information from "MegaMath" which is made available under the ODC Attribution License.

### 13. FineWeb2
A high quality pretraining data to over 1000 languages.

- **Source:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- **License:** Open Data Commons Attribution License v1.0 (ODC-By 1.0)
- **Copyright:** @ Hugging Face Science team
> Licensed under the Open Data Commons Attribution License v1.0.
> To view a copy of this license, visit https://opendatacommons.org/licenses/by/1.0/
>
> **Attribution:** This project contains information from "FineWeb2" which is made available under the ODC Attribution License.

### 14. FineWeb2-HQ
A high-quality, model-filtered pretraining dataset derived as a subset of FineWeb2, spanning 20 languages.

- **Source:** https://huggingface.co/datasets/epfml/FineWeb2-HQ
- **License:** Open Data Commons Attribution License v1.0 (ODC-By 1.0)
- **Copyright:** @ EPFL Machine Learning and Optimization Laboratory
> Licensed under the Open Data Commons Attribution License v1.0.
> To view a copy of this license, visit https://opendatacommons.org/licenses/by/1.0/
>
> **Attribution:** This project contains information from "FineWeb2-HQ" which is made available under the ODC Attribution License.

### 15. FineMath
A dataset that consists of 34B tokens (FineMath-3+) and 54B tokens (FineMath-3+ with InfiMM-WebMath-3+) of mathematical educational content filtered from CommonCrawl.

- **Source:** https://huggingface.co/datasets/HuggingFaceTB/finemath
- **License:** Open Data Commons Attribution License v1.0 (ODC-By 1.0)
- **Copyright:** @ 2025 Hugging Face (HuggingFaceTB)
> Licensed under the Open Data Commons Attribution License v1.0.
> To view a copy of this license, visit https://opendatacommons.org/licenses/by/1.0/
>
> **Attribution:** This project contains information from "FineMath" which is made available under the ODC Attribution License.

---
