# ASMT-001: Reasoning Generation Pipeline for Math QA

A robust pipeline for deduplicating math questions and generating step-by-step reasoning in Korean using OpenAI's GPT models. This project is designed to clean math datasets, generate high-quality explanations, and verify their correctness.

## 🚀 Overview

This pipeline processes a dataset of math questions and answers (`dataset.csv`) through the following stages:

1. **Deduplication**: Identifies and removes duplicate questions using embedding similarity and LLM verification.
2. **Reasoning Generation**: Generates detailed, step-by-step reasoning for each unique question.
3. **Verification**: Validates the generated reasoning by extracting the final answer and comparing it with the ground truth.

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **LLM**: OpenAI API (GPT-4 or equivalent for reasoning and embeddings)
- **Libraries**: pandas, scikit-learn (cosine similarity), tqdm, openai, python-dotenv

## ⚙️ Installation & Setup

### 1. Prerequisites

Ensure you have `uv` installed. If not, you can install it via:

```bash
curl -LsSf https://astral-sh.uv.re/install.sh | sh
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Environment Variables

Create a `.env` file from the template and add your OpenAI API key:

```bash
cp .env.template .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 📂 Data Formats

### Input: `dataset.csv`

| Column     | Description                         |
| ---------- | ----------------------------------- |
| `question` | The math problem statement (Korean) |
| `answer`   | The ground truth answer             |

### Output Files

1.  **`duplicates.csv`**: Questions identified as duplicates.
    - Columns: `question`, `duplicated_from`, `reason`
2.  **`correct_dataset.csv`**: Questions where the LLM's reasoning led to the correct answer.
    - Columns: `question`, `answer`, `reasoning`
3.  **`incorrect_dataset.csv`**: Questions where the LLM's reasoning failed or produced the wrong answer.
    - Columns: `question`, `answer`, `reasoning`

## 🏃 Usage

To run the entire pipeline:

```bash
uv run main.py
```

## 🏗 Project Structure

```text
.
├── src/
│   ├── client.py          # OpenAI API client configuration
│   ├── data_manager.py    # CSV loading and saving
│   ├── deduplication.py   # Embedding-based similarity check
│   ├── embedding.py       # OpenAI embedding generation
│   ├── preprocessing.py   # Question canonicalization
│   ├── reasoning.py       # Step-by-step reasoning generation
│   └── verification.py    # Answer extraction and comparison
├── main.py                # Pipeline entry point
├── pyproject.toml         # Dependencies and project metadata
└── README.md              # Project documentation
```

## 🔍 Pipeline Details

1. **Preprocessing**: Questions are canonicalized to reduce superficial differences
2. **Embedding Generation**: Embedding vectors are generated for each question.
3. **Duplicate Detection**: Detect duplicated questions.
   - Pairs with cosine similarity > 0.9 are flagged as potential duplicates.
   - LLM verification is used to confirm if they are true duplicates (e.g., different phrasing of the same question).
4. **Reasoning Generation**: For each unique question, the LLM generates a step-by-step reasoning process.
   - Answers are not given to the LLM, so that it must derive the answer through reasoning.
5. **Verification**: The final answer is extracted from the generated reasoning and compared to the ground truth answer. The results are categorized into correct and incorrect datasets.
   - In actuall implementation, you can probably make LLM retry the **reasoning generation** if the answer is incorrect, up to a certain number of attempts. For this assignment, we will just categorize it as incorrect without retrying.
