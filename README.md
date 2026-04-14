# 🧠 Medical Reasoning Graph Generator

A pipeline that converts medical reasoning text into a structured, validated knowledge graph using LLMs, SciSpacy, and ontology constraints.

---

##  Features

*  Converts reasoning into **atomic claims**
*  Transforms claims into **Subject–Predicate–Object triplets**
*  Classifies entities using **UMLS + SciSpacy**
*  Validates relations using a **custom ontology**
*  Generates a **visual reasoning graph**
*  Fully **local LLM pipeline** using Ollama

---



##  Project Structure

```
medical-verifier/
│
├── atomic.py              # Atomic claim extraction
├── spo_graph.py           # Triplet generation + graph building
├── ontology.py            # Ontology rules (node + edge constraints)
├── graph_approach.py      # Graph visualization
├── main.py                # Main entry point
│
├── setup.bat              # One-time setup script
├── run.bat                # Run pipeline
├── start_ollama.bat       # Start Ollama server
│
└── reasoning_graphs/      # Output graphs
```

---

##  Requirements

* Python **3.10**
* Ollama (local LLM runtime)

---

##  Setup Instructions

### 1. Install Python 3.10

Download from: https://www.python.org/downloads/

---

### 2. Install Ollama

Download from: https://ollama.com

---

### 3. Run Setup

Double-click:

```
setup.bat
```

This will:

* Create virtual environment
* Install dependencies
* Install SciSpacy model

---

### 4. Download LLM model

Run:

```
ollama pull phi4
```

---

## ▶️ Running the Project

### Step 1: Start Ollama server

```
start_ollama.bat
```

---

### Step 2: Run pipeline

```
run.bat
```

---

## Alternative (Manual Run)

```bash
venv\Scripts\activate
ollama serve
python main.py --input_file input.txt --visualize
```

---

## Input Format

You can either:

### Option 1: CLI input

Run:

```
python main.py
```

Then paste reasoning text.

---

### Option 2: File input (recommended)

Create `input.txt`:

```
Hypertension increases blood pressure.
High blood pressure damages blood vessels.
```

Run:

```bash
python main.py --input_file input.txt --visualize
```

---

## Output

* JSON graph saved in:

```
reasoning_graphs/
```

* Graph visualization:

```
*.png
```

---

## Notes

* First run may download ~500MB (UMLS embeddings)
* Ollama must be running (`ollama serve`)
* Default model: `phi4`

