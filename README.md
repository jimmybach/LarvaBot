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

## Hugging Face Authentication

Some models (such as Llama 3.2 Instruct and private Hugging Face repositories) require authentication before they can be downloaded.

### Local Development

Create a file named `secrets.toml` inside a `.streamlit` directory at the project root:

```text
LarvaBot/
├── .streamlit/
│   └── secrets.toml
├── Larva_Bot/
├── requirements.txt
└── README.md
```

Add your Hugging Face token:

```toml
HF_TOKEN = "hf_your_token_here"
```

Access the token within the application:

```python
import streamlit as st

hf_token = st.secrets["HF_TOKEN"]
```

Pass the token when loading models:

```python
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token
)
```

### Streamlit Community Cloud

For deployments on Streamlit Community Cloud:

1. Open the application settings.
2. Navigate to **Secrets**.
3. Add the following entry:

```toml
HF_TOKEN = "hf_your_token_here"
```

The same application code will automatically access the secret through `st.secrets`.

### Obtaining a Hugging Face Token

1. Create a Hugging Face account.
2. Navigate to **Settings → Access Tokens**.
3. Create a new token with **Read** permissions.
4. Copy the generated token into your local `secrets.toml` file or Streamlit Cloud Secrets configuration.

### Security Notes

* Never commit `secrets.toml` to GitHub.
* Add the following entry to `.gitignore`:

```text
.streamlit/secrets.toml
```

* If a token is accidentally exposed, revoke it immediately and generate a new one through Hugging Face.

### Accessing Gated Models

Certain models, including Meta's Llama family, require acceptance of a model license before download.

To request access:

1. Visit the model page on Hugging Face.
2. Accept the license agreement.
3. Ensure the account associated with your access token has been granted access.
4. Restart the application after access has been approved.


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
