python -m venv .venv

source .venv/bin/activate     # macOS/Linux

.venv\Scripts\activate        # Windows

Do this:
`
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
`

And this:
`
pip install -r requirements.txt
`