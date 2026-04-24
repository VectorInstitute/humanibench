# HumaniBench: A Human-Centric Benchmark for Large Multimodal Models Evaluation

--------------------------------

[![code checks](https://github.com/VectorInstitute/HumaniBench/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/HumaniBench/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/HumaniBench/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/HumaniBench/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/HumaniBench/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/HumaniBench/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/HumaniBench/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/HumaniBench/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/HumaniBench/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/HumaniBench)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/HumaniBench)

<p align="center">
  <img src="https://github.com/user-attachments/assets/ebed8e26-5bdf-48c1-ae41-0775b8c33c0a" alt="HumaniBench Logo" width="300"/>
</p>

<p align="center">
  <b>🌐 Website:</b> <a href="https://vectorinstitute.github.io/humanibench/">vectorinstitute.github.io/humanibench</a>
  &nbsp;|&nbsp;
  <b>📄 Paper:</b> <a href="https://arxiv.org/abs/2505.11454">arxiv.org/abs/2505.11454</a>
  &nbsp;|&nbsp;
  <b>📊 Dataset:</b> <a href="https://huggingface.co/datasets/vector-institute/HumaniBench">Hugging Face</a>
</p>

---


## 🧠 Overview

As multimodal generative AI systems become increasingly integrated into human-centered applications, evaluating their **alignment with human values** has become critical.

**HumaniBench** is the **first comprehensive benchmark** designed to evaluate **Large Multimodal Models (LMMs)** on **seven Human-Centered AI (HCAI) principles**:

* **Fairness**
* **Ethics**
* **Understanding**
* **Reasoning**
* **Language Inclusivity**
* **Empathy**
* **Robustness**

This repository provides code and scripts for evaluating LMMs across **7 human-aligned tasks**.

---

## 📦 Features

* 📷 **32,000+ Real-World Image–Question Pairs**
* ✅ **Human-Verified Ground Truth Annotations**
* 🌐 **Multilingual QA Support (10+ languages)**
* 🧠 **Open and Closed-Ended VQA Formats**
* 🧪 **Visual Robustness & Bias Stress Testing**
* 📑 **Chain-of-Thought Reasoning + Perceptual Grounding**


---

## 📂 Evaluation Tasks Overview

| Task                               | Focus                                                                                          | Folder                             |
| :--------------------------------- | :--------------------------------------------------------------------------------------------- | :--------------------------------- |
| **Task 1: Scene Understanding**    | Visual reasoning + bias/toxicity analysis in social attributes (gender, age, occupation, etc.) | `src/task1_scene_understanding`   |
| **Task 2: Instance Identity**      | Visual reasoning in culturally rich, socially grounded settings                                | `src/task2_instance_identity`     |
| **Task 3: Multiple Choice QA**     | Structured attribute recognition via multi-choice questions                                    | `src/task3_multiplechoice_vqa`    |
| **Task 4: Multilingual Visual QA** | VQA across 10+ languages, including low-resource ones                                          | `src/task4_multilingual`          |
| **Task 5: Visual Grounding**       | Bounding box localization of socially salient regions                                          | `src/task5_visual_grounding`      |
| **Task 6: Empathetic Captioning**  | Human-style emotional captioning evaluation                                                    | `src/task6_empathetic_captioning` |
| **Task 7: Image Resilience**       | Robustness testing via image perturbations                                                     | `src/task7_image_resilience`      |

> 🔍 Each task folder includes a README with setup instructions, task structure, and metrics.

---

## 🧬 Pipeline

**Three-stage process:**

1. **Data Collection**
   Curated from global news imagery, tagged by social attributes (age, gender, race, occupation, sport)

2. **Annotation**
   GPT-4o–assisted labeling + human expert verification

3. **Evaluation**
   Comprehensive scoring across:

   * Accuracy
   * Fairness
   * Robustness
   * Empathy
   * Faithfulness

---

## 🔑 Key Insights

* 🔍 **Bias persists**, especially across gender and race
* 🌐 **Multilingual gaps** affect low-resource language performance
* ❤️ **Empathy and ethics** vary significantly by model family
* 🧠 **Chain-of-Thought reasoning** improves performance but doesn’t fully mitigate bias
* 🧪 **Robustness tests** reveal fragility to noise, occlusion, and blur

---

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```

If you're coming from `poetry` then you'll notice that the virtual environment
is actually stored in the project root folder and is by default named as `.venv`.
The other important note is that while `poetry` uses a "flat" layout of the project,
`uv` opts for the the "src" layout. (For more info, see [here](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/))

---

## 📚 Citation

If you use HumaniBench or this evaluation suite in your work, please cite:

```bibtex
        @article{raza2025humanibench,
            title={Humanibench: A human-centric framework for large multimodal models evaluation},
            author={Raza, Shaina and Narayanan, Aravind and Khazaie, Vahid Reza and Vayani, Ashmal and Radwan, Ahmed Y and Chettiar, Mukund S and Singh, Amandeep and Shah, Mubarak and Pandya, Deval},
            journal={arXiv preprint arXiv:2505.11454},
            year={2025}
          }

```
---
## 🙏 Acknowledgments

Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute ([vectorinstitute.ai/#partners](http://www.vectorinstitute.ai/#partners)).

This research was funded by the European Union's Horizon Europe research and innovation programme under the AIXPERT project (Grant Agreement No. 101214389), which aims to develop an agentic, multi-layered, GenAI-powered framework for creating explainable, accountable, and transparent AI systems.


---

## 📬 Contact

For questions, collaborations, or dataset access requests, please [open an issue](https://github.com/VectorInstitute/humanibench/issues) in this repository or contact the corresponding author at [shaina.raza@vectorinstitute.ai](mailto:shaina.raza@vectorinstitute.ai), as listed in the paper.

---

### ⚡ HumaniBench promotes trustworthy, fair, and human-centered multimodal AI.

**We invite researchers, developers, and policymakers to explore, evaluate, and extend HumaniBench. 🚀**

---
