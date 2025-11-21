---
pretty_name: EMS-MCQA
dataset_name: EMS-MCQA
task_categories:
  - question-answering
language:
  - en
size_categories:
  - n<10K
license: mit
---

# EMS-MCQA

**EMS-MCQA** is a multiple-choice question answering (MCQA) dataset focused on
Emergency Medical Services (EMS) knowledge. See more on our [project page](https://uva-dsa.github.io/EMSQA/).


This repo contains an **open-source subset** of the full dataset, provided as
JSON files.

---

## Dataset Summary

Each record is a JSON object with:

- **`question`** *(str)* – question stem  
- **`choices`** *(list[str])* – options (letters like `a.`, `b.`, … are kept)  
- **`answer`** *(str)* or *(list[str])* – correct option label (`"a" | "b" | ["a", "c"]`)  
- **`explanation`** *(str, optional)* – short rationale  
- **`link`** *(str, optional)* – source URL  
- **`level`** *(list[str])* – EMS certification level(s), e.g., `emr`, `emt`, `aemt`, `paramedic`, `NA`  
- **`category`** *(list[str])* – topical tags (e.g., `airway_respiration_and_ventilation`, `anatomy`, `assessment`, `cardiology_and_resuscitation`, `ems_operations`, `medical_and_obstetrics_gynecology`,
`pediatrics`, `pharmacology`, `trauma`)

---

## Files & Counts

- `train_open.json` — **13,021** items  
- `val_open.json` — **1,860** items  
- `test_open.json` — **3,721** items  
- `MCQA_open_final.json` — **18,602** items (entire open-source collection)  
- `train_open_woNA.json` — training split with items where `level != "NA"`

> Use `train_open_woNA.json` when you want to predict certification/subject without unlabeled levels.

---

## Citation

If you use EMS-MCQA in your work, please cite:

```bibtex
@misc{ge2025expertguidedpromptingretrievalaugmentedgeneration,
  title         = {Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering},
  author        = {Xueren Ge and Sahil Murtaza and Anthony Cortez and Homa Alemzadeh},
  year          = {2025},
  eprint        = {2511.10900},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2511.10900},
}
```

---

## Loading with 🤗 Datasets

```python
from datasets import load_dataset

# Load predefined splits from this repo
ds = load_dataset("Xueren/EMS-MCQA")       # returns a DatasetDict with train/validation/test
print("train:", ds["train"][0])

# Or load the full open-source collection as a single split
full_data   = load_dataset("Xueren/EMS-MCQA", split="full_data")
print("full_data:", full_data[0])

# Or load the train without NA certification label
train_woNA = load_dataset("Xueren/EMS-MCQA", split="train_woNA")
print("train_woNA:", train_woNA[0])