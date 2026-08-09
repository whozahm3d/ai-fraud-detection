# Contributing to AI Fraud Detection

Thank you for your interest in contributing! 🚀
This repository is an AI system focused on **financial fraud detection using Machine Learning, Explainable AI, and Retrieval-Augmented Generation (RAG)**.

---

## 📌 How You Can Contribute

- Improve data preprocessing scripts or feature engineering
- Optimise machine learning models or add new classifiers
- Enhance XAI / model interpretability (LIME, Integrated Gradients, etc.)
- Improve the RAG policy retrieval system (new documents, better chunking)
- Add Docker / containerisation support
- Improve data visualisations and EDA
- Write unit tests for the preprocessing pipeline
- Fix bugs or improve performance
- Improve documentation

---

## 🛠️ Development Setup

### 1. Fork and clone

```bash
git clone https://github.com/whozahm3d/ai-fraud-detection.git
cd ai-fraud-detection
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Runtime dependencies
pip install -r requirements.txt

# Development dependencies (linting, notebooks, testing)
pip install -r requirements-dev.txt
```

Or using Make:

```bash
make install-dev
```

### 4. Create a feature branch

```bash
git checkout -b feature/your-feature-name
```

---

## 📋 Contribution Guidelines

- Use **clear and descriptive commit messages** (e.g. `feat: add LIME explanations`, `fix: handle empty CSV upload`)
- Follow **PEP 8 Python style guidelines** — run `make lint` before submitting
- Comment important or non-obvious logic
- **Do not upload large dataset files** — `Data/original_dataset/` is in `.gitignore`
- **Do not commit `.streamlit/secrets.toml`** — it contains your API key
- Test your changes before submitting a Pull Request
- Keep the repository clean and organised

---

## 🔍 Code Style

- Follow **PEP 8** — max line length 120 characters
- Use **snake_case** for variable and function names
- Write modular, reusable functions with docstrings on complex logic
- Avoid hardcoded values — use constants or config where possible

Run the linter:

```bash
make lint
# or directly:
flake8 app.py rag_module.py scripts/ --max-line-length=120
```

---

## 🚀 Submitting Changes

1. Push your branch:

```bash
git push origin feature/your-feature-name
```

2. Open a **Pull Request** against `main`
3. Fill in the PR template — describe what changed, why, and how you tested it
4. The maintainer will review and provide feedback

---

## 🧪 Testing

Before submitting:

- Ensure `streamlit run app.py` launches without errors
- Verify the affected dashboard page(s) work as expected
- Confirm notebooks run end-to-end if you modified them
- Run `make lint` and resolve any errors

---

## ⚠️ Security Reminders

- Never commit `.streamlit/secrets.toml`
- Never hardcode API keys, passwords, or credentials in source files
- Never commit large dataset files from `Data/original_dataset/`
- See [SECURITY.md](SECURITY.md) for the full security policy

---

## 📄 License

By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

Thank you for helping improve AI Fraud Detection! 🙌
