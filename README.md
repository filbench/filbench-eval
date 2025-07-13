<a href="https://github.com/filbench"><img src="https://avatars.githubusercontent.com/u/194051464?s=400&u=9e2248f7dddd79407badb14a56ebc6a7e773ff26&v=4" width="125" height="125" align="right" /></a>

# FilBench: Open LLM Eval Suite for Filipino

<p align="left">
<b><a href="https://huggingface.co/spaces/UD-Filipino/filbench-leaderboard">🥇 Leaderboard</a></b>
|
<b><a href="https://github.com/filbench/lighteval">💻 Evaluation Runner (Lighteval)</a></b>
|
<b><a href="">📄 Paper</a></b>
</p>

This repository contains all relevant tools and experiments for FilBench, an Open LLM Evaluation Suite and leaderboard for Filipino.
We curated an evaluation benchmark consisting of four major capabilities: Cultural Knowledge, Generation, Reading Comprehension, and Classical NLP.
Then, we evaluated over 20 models of different parameters, model families, and multilingual support.

## 📰 News

- [2025-08-08] FilBench is now an official part of HuggingFace's Community Tasks in [Lighteval](https://github.com/huggingface/lighteval)! You can also read the corresponding blog post [here]().
- [2025-08-01] We officially introduce FilBench! You can read more details in our [report]().

## 🔧 Installation

The installation process assumes you have [uv](https://docs.astral.sh/uv/) and Python 3.12 installed.
First, clone this repository and install all dependencies and our fork of `lighteval`:

```sh
git clone git@github.com:filbench/filbench-eval.git
cd filbench-eval
git submodule update
uv sync
```

These steps will clone our `lighteval` fork as a submodule and install necessary dependencies.
In the end, you should have access to the following tools:

1. **lighteval**: this is the main evaluation runner to use for launching evaluation jobs.
2. **filbench**: a lightweight CLI for computing the FilBench score and reporting results to the leaderboard.

You can check if your installation works by running the following commands:

```sh
# Check if filbench installation works
filbench --help

# Check if lighteval installation works
cd lighteval
python3 -m lighteval tasks inspect "filbench|cebuaner_ceb_mcf|0|0" \
    --num-samples 1 \
    --custom-tasks community_tasks/filbench_evals.py
```

> [!IMPORTANT]
> You must run the `lighteval` command (1) within the `lighteval` submodule and (2) using the `python -m ...` prefix. If you encounter any installation issues, please [open an Issue](https://github.com/filbench/filbench-eval/issues/new) in this repository.

## 👩‍💻 Usage

### Running evaluations on FilBench

In order to run all evaluations on FilBench, we recommend running the following command:

```sh
cd lighteval
export HF_ORG=<...>
# For models in HuggingFace and accessible via vLLM
cat examples/tasks/all_filbench_tasks.txt | xargs -I {} \
    python -m lighteval vllm "pretrained=<MODEL_NAME>" {} \
    --push-to-hub \
    --results-org $HF_ORG \
    --custom-tasks community_tasks/filbench_evals.py
```

This command will then run all tasks in FilBench on `MODEL_NAME`, and upload the results to `HF_ORG`.
When run in parallel, the shortest task can take around 5 minutes and the longest task can take around 2 hours.

### Computing the FilBench Score

Your results should be saved in `HF_ORG/MODEL_NAME`.
For example, the results for aisingapore/SEA-LION-v3.5-70B-R are stored in [`UD-Filipino/details_aisingapore__Llama-SEA-LION-v3.5-70B-R_private`](https://huggingface.co/datasets/UD-Filipino/details_aisingapore__Llama-SEA-LION-v3.5-70B-R_private).
To compute the FilBench score, run the following command:

```sh
filbench compute-score <HF_ORG>/<MODEL_NAME>
# For example:
filbench compute-score UD-Filipino/details_aisingapore__Llama-SEA-LION-v3.5-70B-R_private
```

### Submitting to the Leaderboard

We also maintain a [leaderboard](https://huggingface.co/spaces/UD-Filipino/filbench-leaderboard) to track the progress in Filipino NLP.
By default, this command will output a JSON file called `scores_<HF_ORG>___<MODEL_NAME>.json` that contains the FilBench score and its breakdown across categories and tasks.
You can then submit these results by running the command below and following the prompts:

```sh
filbench submit "scores_<HF_ORG>__MODEL_NAME.json"
# 🤗 Model Name or HuggingFace ID (e.g., Qwen/Qwen3-32B):
# ...
```

This will then make a PR to the `UD-Filipino/filbench-results-submission` dataset.
The approval process is done manually, and we might contact you to clarify a few things.

> [!NOTE]
> If you want to update your scores for a specific model in the leaderboard, you just need to rerun the `submit` command and input the **same** organization and model name.
> Internally, we hash these variables together and show the latest result.

> [!TIP]
> You can set the `--dry-run` flag to double-check whether the details you entered are correct.

## 📜 Team

This work was done by [@ljvmiranda921](https://github.com/ljvmiranda921), [@elyanah-aco](https://github.com/elyanah-aco), [@connermanuel](https://github.com/connermanuel), [@jcblaisecruz02](https://github.com/jcblaisecruz02), and [@imperialite](https://github.com/imperialite). 
For any questions, please reach out to us via filbench-eval@googlegroups.com or through our [GitHub Issues](https://github.com/filbench/filbench-eval/issues).
To cite our work, please use the following bibTeX entry:

```bib
@article{filbench,
  title={Fil{B}ench: {C}an {LLM}s {U}nderstand and {G}enerate {F}ilipino?},
  author={Miranda, Lester James V and Aco, Elyanah and Manuel, Conner and Cruz, Jan Christian Blaise and Imperial, Joseph Marvin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🙏 Acknowledgments

We would like to thank [Cohere Labs](https://cohere.com/research) for providing credits through the [Cohere Research Grant](https://cohere.com/research/grants) to run the Aya model series, and [Together AI](https://together.ai) for additional computational credits for running several open models.
We also acknowledge the Hugging Face team, particularly the OpenEvals team ( [@clefourrier](https://github.com/clefourrier) and [@NathanHB](https://github.com/NathanHB)) and [@davanstrien](https://github.com/davanstrien), for their support in publishing the FilBench blog post.
