# Installation

## Prerequisites

* Python 3.12+
* Git
* Hugging Face account (required for gated models such as Llama 3.2)

Verify your Python version:

```bash
python --version
```

## Clone the Repository

```bash
git clone https://github.com/jimmybach33/larvabot.git
cd larvabot
```

## Create a Virtual Environment

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Hugging Face Authentication (Optional)

Some models require authentication through Hugging Face.

Login locally:

```bash
huggingface-cli login
```

Or set an access token:

```bash
export HF_TOKEN="your_token_here"
```

On Windows:

```powershell
$env:HF_TOKEN="your_token_here"
```

## Running the Application

From the repository root:

```bash
streamlit run Larva_Bot/app/larva_app_LOCAL.py
```

The application will be available at:

```text
http://localhost:8501
```

## Model Options

### Qwen 0.6B

Recommended for:

* Fast local inference
* Lower memory usage
* Development and testing

### LarvaBot 4B Fine-Tuned Model

Recommended for:

* Higher response quality
* More accurate personality replication

Requirements:

* Significantly more RAM
* Longer response generation times

## Troubleshooting

### No module named 'src'

Ensure you are running Streamlit from the repository root:

```bash
cd larvabot
streamlit run Larva_Bot/app/larva_app_LOCAL.py
```

### No module named 'chromadb'

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Tokenizer Errors

Install tokenizer dependencies:

```bash
pip install sentencepiece tiktoken protobuf
```

### Model Loading Issues

Verify that:

* The model path exists locally, or
* The Hugging Face repository name is correct
* Required Hugging Face permissions have been granted

### Slow Generation

The 4B model may take several seconds to generate responses on consumer hardware. For faster responses, use the Qwen 0.6B option.
