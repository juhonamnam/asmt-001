# ASMT-001: Deduplication and Reasoning Generation Pipeline for Math QA Datasets

## Project Overview

This project implements a **Reasoning Generation Pipeline for Math QA datasets**.

The goal is to read a dataset containing math questions and answers, remove duplicated questions, generate step-by-step reasoning using an LLM, verify the generated answers against the ground truth, and produce a new dataset.

The pipeline will run **once** when executing `main.py`.

The dataset is written in **Korean**, and the LLM is expected to generate reasoning in Korean as well.

---

## Tech Stack

Language: Python  
Package Manager: uv

LLM Provider: OpenAI API

Environment variables should be loaded from `.env`.

---

## Input Dataset

File name: `dataset.csv`

CSV columns:
| column | name |
|--------|----------|
| 1 | question |
| 2 | answer |

Example:
question,answer
"2x + 5 = 9일 때 x의 값을 구하시오","2"
"반지름이 3인 원의 넓이는?","9π"

---

## Output Dataset

The pipeline generates:


- `duplicates.csv` containing duplicated questions with the following columns:
  | column | name |
  |--------|-----------|
  | 1 | question |
  | 2 | dulpicated_from |
  | 3 | reason |

  Example:
  question,dulpicated_from,reason
  "2x + 5 = 9일 때 x의 값을 구하시오","x를 구하시오: 2x + 5 = 9","High embedding similarity + LLM confirmation"

- `correct_dataset.csv` containing the following columns:
  | column | name |
  |--------|-----------|
  | 1 | question |
  | 2 | answer |
  | 3 | reasoning |

  This dataset contains questions where the generated reasoning leads to the correct answer.

  Example:
  question,answer,reasoning
  "2x + 5 = 9일 때 x의 값을 구하시오","2","1. 양변에서 5를 뺍니다: 2x = 4\n2. 양변을 2로 나눕니다: x = 2"
  "반지름이 3인 원의 넓이는?","9π","1. 원의 넓이 공식은 A = πr^2입니다.\n2. 반지름 r이 3이므로, A = π \* 3^2 = 9π입니다."

- `incorrect_dataset.csv` containing the following columns:
  | column | name |
  |--------|-----------|
  | 1 | question |
  | 2 | answer |
  | 3 | reasoning |

  This dataset contains questions where the generated reasoning does not lead to the correct answer.

  Example:
  question,answer,reasoning
  "2x + 5 = 9일 때 x의 값을 구하시오","3","1. 양변에서 5를 뺍니다: 2x = 4\n2. 양변을 2로 나눕니다: x = 2"

---

## Steps in the Pipeline

1. **Load Dataset**: Read `dataset.csv` into a DataFrame.
2. **Preprocess Dataset**: Canonicalize questions. (Rule based + LLM based)
3. **Embedding Generation**: Generate embeddings for question/answer pairs using OpenAI API.
4. **Deduplication**: Remove duplicate questions based on cosine similarity of embeddings.
   - If similarity > 0.9, use LLM to determine if questions are actually duplicates.
5. **Reasoning Generation**: For each unique question, generate step-by-step reasoning using the LLM.
6. **Answer Verification**: Verify the generated answer against the ground truth answer.
7. **Output Dataset**: Save the results to `output_dataset.csv`.
