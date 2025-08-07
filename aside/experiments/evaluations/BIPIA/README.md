# BIPIA

[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightblue?style=flat-square)](https://arxiv.org/abs/2312.14197)
[![Code](https://img.shields.io/badge/Code-%F0%9F%92%8e-lightgrey?style=flat-square)](https://github.com/microsoft/BIPIA)


The data and code of our work "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models".
If you believe that the content on this repo infringes your rights, please contact us for requesting a take down.


## Overview
Recent advancements in large language models (LLMs) have led to their adoption across various applications, notably in combining LLMs with external content to generate responses. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.

We introduce the first **b**enchmark of **i**ndirect **p**rompt **i**njection **a**ttack, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. We also propose several defenses for both black-box and white-box scenarios. We hope that our benchmark and defenses can inspire future work in this important area.


## Requirements

### Software requirements
Install bipia and its dependencies from source:
```bash
git clone git@github.com:microsoft/BIPIA.git
pip install .
```

The package has been tested and verified to work on Linux: Ubuntu 20.04.6. It is recommended to use this operating system for optimal compatibility.


### Hardware requirements
For the evaluation of the robustness of LLMs to indirect prompt injection attacks, we recommend using a machine with the following specifications:
1. For experiments related to API-based models (such as GPT), you can complete them on a machine without a GPU. However, you will need to set up an account's API key.
2. For open-source models of 13B and below, our code has been tested on a machine with 2 V100 GPUs. For models larger than 13B, 4-8 V100 GPUs are required. If there are GPUs with better performance, such as A100 or H100, you can also use them to complete the experiments. Fine-tuning-based experiments are completed on a machine with 8 V100 GPUs.




## How to use
We provide a simple example in [demo.ipynb](demo.ipynb) to demonstrate how to use the code to load the dataset and evaluate the robustness of LLMs to indirect prompt injection attacks.

### Download the dataset
<!-- In our work, we realse the first **b**enchmark of **i**ndirect **p**rompt **i**njection attack, named BIPIA.
There are two methods to load the dataset. -->

<!-- - Load dataset from huggingface:
```python
from datasets import load_dataset

dataset = load_dataset("bipia", dataset_name)
``` -->

Load BIPIA dataset with the following python script:
```Python
from bipia import AutoPIABuilder

pia_builder = AutoPIABuilder.from_name(dataset_name)(seed=2023)
pia_samples = pia_builder(
    context_data_file,
    attack_data_file,
    enable_stealth=False,
)
pia_dataset = Dataset.from_pandas(pia_samples)
```

For different task of different split (train/test), set `context_data_file` as the files in `benchmark/{task}/{train|test}.jsonl` directory.  set `attack_data_file` as `benchmark/{code|text}_attack_{train|test}.json`. The configureation of `dataset_name` is as follows:
- EmailQA: set `dataset_name` as `email`
- WebQA: set `dataset_name` as `qa`
- Summarization: set `dataset_name` as `abstract`
- TableQA: set `dataset_name` as `table`
- CodeQA: set `dataset_name` as `code`

*Note: For Summarization and WebQA task, due to license issues, please follow the guidelines in [benchmark/README.md](benchmark/README.md) to generate `context_data_file`.*



#### Evaluation
In our work, we evaluate the robustness of 25 existing large language models to indirect prompt injection attacks on BIPIA.
To reproduce the evaluation results in our paper, execute the following commands.

```bash
cd examples

# generate respones
python run.py --seed 2023 --dataset_name {task} \
--context_data_file path/of/external/conten/file \
--attack_data_file path/of/attack/file \
--llm_config_file config/{llm_name}.yaml \
--batch_size 20 --output_path path/of/output/file \
--log_steps 10 --resume

# compute attack success rate
python run.py --mode evaluate --seed 2023 \
--dataset_name {task} \
--response_path path/of/output/file \
--output_path path/of/asr/file \
--gpt_config_file config/{evaluate_llm_name}.yaml \
--batch_size 20 --log_steps 10 --resume
```

Arguments:
- `task`: the selected task name, you can choose anyone from `["code", "email", "qa", "abstract", "table"]`
- `llm_name`: the name of the LLMs. Select from the config file in `config` directory.
- `evaluate_llm_name`: the name of the LLMs for evaluation. Use `gpt35` by default.

### Defense
We also propose two type of defense methods.

- Meta-prompting Defenses
  - Border Strings
  - In-context Learning
  - Multi-turn Dialogue

- Finetuning Defenses
  - Speical Tokens

Meanwhile, we relase our defense code for reproducing our results. 

See instructions for running defense at [defense/bipia_defense](defense/README.md).

## License
This project is licensed under the license found in the [LICENSE](https://github.com/microsoft/BIPIA/blob/main/LICENSE) file in the root directory of this source tree. Portions of the source code are based on the evaluate project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


## Responsible AI Transparency Information 

An AI system includes not only the technology, but also the people who will use it, the people who will be affected by it, and the environment in which it is deployed. Creating a system that is fit for its intended purpose requires an understanding of how the technology works, its capabilities and limitations, and how to achieve the best performance. Microsoft has a broad effort to put our AI principles into practice. To find out more, see [Responsible AI principles from Microsoft](https://www.microsoft.com/en-us/ai/responsible-ai).


### Use of this code

Our goal in publishing this code is to facilitate reproducibility of our paper in hopes of motivating further research in defending against indirect prompt injection attacks. 
Our goal is to enhance the reliable and secure utilization of powerful LLMs and to inspire further research on this crucial issue.
This code should only be used for research on indirect prompt injection attacks.


### Project data 

This project includes a dataset with five tasks: Web QA, Email QA, Table QA, Summarization, and Code QA. The data for Email QA comes from [OpenAI Evals](https://github.com/openai/evals), for Table QA from [WikiTableQuestions](https://arxiv.org/abs/1508.00305), and for Code QA from [Stack Exchange](https://archive.org/details/stackexchange).
The datasets for Web QA and Summarization, due to license reasons, require users to read the usage terms of the source data and download it from the source, then process the data using the scripts we provide. 
For more information, see `Download the dataset` in the `How to use` section below.


### Fairness and Responsible AI testing

At Microsoft, we strive to empower every person on the planet to do more. An essential part of this goal is working to create technologies and products that are fair and inclusive. Fairness is a multi-dimensional, sociotechnical topic and impacts many different aspects of our work.  

When systems are deployed, Responsible AI testing should be performed to ensure safe and fair operation for the specific use case. No Responsible AI testing has been done to evaluate this method including validating fair outcomes across different groups of people. Responsible AI testing should be done before using this code in any production scenario. 

> Note: The documentation included in this ReadMe file is for informational purposes only and is not intended to supersede the applicable license terms. 


## TODO
- [ ] Upload the BIPIA dataset to huggingface.
- [ ] Release all responses of evaluated LLMs.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Paper
If you use this code in your research please cite the following [publication](https://arxiv.org/abs/2312.14197):
```
@article{yi2023benchmarking,
  title={Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models},
  author={Yi, Jingwei and Xie, Yueqi and Zhu, Bin and Hines, Keegan and Kiciman, Emre and Sun, Guangzhong and Xie, Xing and Wu, Fangzhao},
  journal={arXiv preprint arXiv:2312.14197},
  year={2023}
}
```
