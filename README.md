# JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models

<div align="center">

[[Website]](http://craftjarvis-jarvis1.github.io/)
[[Paper]](https://arxiv.org/abs/2311.05997)
[[Twitter]](https://twitter.com/jeasinema/status/1723900032653643796)

![](assets/jarvis.gif)

</div>

## Abstract 

Achieving human-like planning and control with multimodal observations in an open world is a key milestone for more functional generalist agents. We introduce **JARVIS-1**, an open-world agent that can perceive multimodal input (visual observations and human instructions), generate sophisticated plans, and perform embodied control, all within the popular yet challenging open-world Minecraft universe. Specifically, we develop **JARVIS-1** on top of pre-trained multimodal language models, which map visual observations and textual instructions to plans. The plans will be ultimately dispatched to the goal-conditioned controllers. We outfit **JARVIS-1** with a multimodal memory, which facilitates planning using both pre-trained knowledge and its actual game survival experiences. **JARVIS-1** is the existing most general agent in Minecraft, capable of completing over 200 different tasks using control and observation space similar to humans. These tasks range from short-horizon tasks, e.g., "chopping trees" to long-horizon tasks, e.g., "obtaining a diamond pickaxe". **JARVIS-1** performs exceptionally well in short-horizon tasks, achieving nearly perfect performance. In the classic long-term task of ObtainDiamondPickaxe, **JARVIS-1** surpasses the reliability of current state-of-the-art agents by 5 times and can successfully complete longer-horizon and more challenging tasks.

## Agent Playing Videos
We list a series of videos showing **JARVIS-1** playing Minecraft. You can find the videos on our [Project Page](http://craftjarvis-jarvis1.github.io/).

## Install Dependencies

This project is intended for running on Linux only. Support for other platforms is not provided.

### Prepare the Environment

We recommand to use Anaconda to manage the environment. If you don't have Anaconda installed, you can download it from [here](https://www.anaconda.com/products/distribution).

```bash
conda create -n jarvis python=3.10
conda activate jarvis 
```

Make sure you have JDK 8 installed. If you don't have it installed, you can install it using the following command:

```bash
conda install openjdk=8
```

To check your JDK version, run the command `java -version`. You should see a message similar to the following (details may vary if you have installed a different JDK distribution):

```bash
openjdk version "1.8.0_392"
OpenJDK Runtime Environment (build 1.8.0_392-8u392-ga-1~20.04-b08)
OpenJDK 64-Bit Server VM (build 25.392-b08, mixed mode)
```

Once you have installed the required dependencies, you can run the `prepare_mcp.py` script to build MCP-Reborn. Make sure you have a stable internet connection before you begin.
```bash
python prepare_mcp.py
```

Then you can install JARVIS-1 as a Python package.
```bash
pip install -e .
```

<!-- <aside>
JARVIS-1 relies on gym==0.23.1, while mineclip and minedojo depend on a different version. If you encounter any errors related to gym versions during installation, you can safely ignore them.
</aside> -->

### Download Weights

We controller rely on the weights of STEVE-I. You can download the weights from the [script](https://github.com/Shalev-Lifshitz/STEVE-1/blob/main/download_weights.sh). 

<!-- Some controller weights from GROOT are not released yet. We will release them in the future. -->

<!-- You also need to download our multimodal memory from the [huggingface link](https://huggingface.co/zhwang4ai/jarvis_memory). -->

## Usage

You need to set the environment variable `TMPDIR` and `OPENAI_API_KEY` first.
```bash
export TMPDIR=/tmp
export OPENAI_API_KEY="sk-******"
```
### Learning with dynamic memory (Coming Soon)

Then you can run the following command to start the JARVIS-1 agent.
```bash
python open_jarvis.py --task iron_pickaxe --timeout 10
```
Finally, you can see the JARVIS-1 agent playing Minecraft in the poped window.
You can also run the following command to start the JARVIS-1 agent in the headless mode.
```bash
xfvb-run -a python open_jarvis.py --task iron_pickaxe --timeout 10
```

### Offline Evaluation with fixed memory

```bash
python offline_evaluation.py
or
xfvb-run -a python offline_evaluation.py
```

<aside>
Now we only release the `offline_evaluation` code, i.e., you can use it to evaluate the JARVIS-1 agent on the fixed memory. We will release the `online_evaluation` code soon, i.e., you can use it to evaluate the JARVIS-1 agent on the growing memory.
</aside>

## Differences from the Original JARVIS-1

- Remove the `self-check` module for efficient planning.
- Current multimodal memory in `assets/memory.json` file is not complete. We remove the multimodal `state` and `action` sequence, which will be released in the future. 
- The `multimodal descriptor` and the `multimodel retrieval` is not released yet. So you can only experience the language model part of JARVIS-1 now.

## To-Do
- [ ] Release `multimodal descriptor` to enable JARVIS-1 to understand the visual world. We plan to upload the `multimodal memory` on huggingface.
- [ ] Release `learning.py` to enables self-improving JARVIS-1 with growing memory.

## Related Projects

**JARVIS-1** is built upon several projects in Minecraft. Here are some related projects that you may be interested in:

- [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1) is an instruction-tuned Video Pretraining (VPT) model for Minecraft. We use it as a part of controller in JARVIS-1.
- [Minedojo](https://github.com/MineDojo/MineDojo) is a simulator suite with 1000s of open-ended and language-prompted tasks built on the popular Minecraft game for embodied agent research.
- [MC-TextWorld](https://github.com/CraftJarvis/MC-TextWorld) is a text world environment for Minecraft. It is designed to be a benchmark for text-based agents. We use it in the early version of JARVIS-1 to accumulate language memory. 

## Check out our paper!
Our paper is available on [Arxiv](https://arxiv.org/pdf/2311.05997.pdf). Please cite our paper if you find **JARVIS-1** useful for your research:
```
@article{wang2023jarvis1,
    title   = {JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models},
    author  = {Zihao Wang and Shaofei Cai and Anji Liu and Yonggang Jin and Jinbing Hou and Bowei Zhang and Haowei Lin and Zhaofeng He and Zilong Zheng and Yaodong Yang and Xiaojian Ma and Yitao Liang},
    year    = {2023},
    journal = {arXiv preprint arXiv: 2311.05997}
}
```
