# Cats vs Dogs Classifier 🐱🐶

A cat-vs-dog image classifier built on a fine-tuned ResNet18,
served as a FastAPI back-end with a Streamlit front-end.

## Run it locally

### 1. Create and activate a virtual environment

From the project root:

    python -m venv .venv

Then activate it (pick the command for your shell):

- Windows (PowerShell):  .venv\Scripts\Activate.ps1
- Windows (cmd.exe):     .venv\Scripts\activate.bat
- macOS / Linux:         source .venv/bin/activate

Your prompt should now start with `(.venv)`.

### 2. Install dependencies

    pip install -r api/requirements.txt
    pip install -r ui/requirements.txt

### 3. Add the trained model weights

Drop `cats_vs_dogs_resnet18.pth` into the `api/` folder.

### 4. Start the API (terminal 1)

    cd api
    uvicorn main:app --reload --port 8000

### 5. Start the UI (terminal 2)

Make sure the venv is activated in this terminal too, then from the project root:

    streamlit run ui/app.py

### 6. Open the app

Visit http://localhost:8501 in your browser.