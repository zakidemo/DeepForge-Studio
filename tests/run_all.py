"""Run every generated export end to end and emit the execution matrix.

This is the functional evaluation the paper reports: for each architecture, the
exported script is run with no edit other than pointing it at data, and the
outcome is recorded. A script that generates but does not train counts as a
failure.

Usage:
    NUMCLASSES=2 INPUTSIZE=64 EPOCHS=1 node tests/harness.mjs
    python3 tests/run_all.py
"""
import os, io, glob, json, shutil, contextlib, traceback
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

GEN = "./generated"
DATA = "./testdata"
N_CLASSES = int(os.environ.get("NUMCLASSES", 2))
IMG = int(os.environ.get("INPUTSIZE", 64))

# --- toy datasets ----------------------------------------------------------
def build_data():
    from PIL import Image
    rng = np.random.default_rng(0)
    if os.path.isdir(DATA):
        shutil.rmtree(DATA)
    # image folders, one per class
    for c in range(N_CLASSES):
        d = f"{DATA}/images/class_{c}"
        os.makedirs(d, exist_ok=True)
        for i in range(8):
            Image.fromarray(rng.integers(0, 255, (IMG, IMG, 3), dtype=np.uint8)).save(f"{d}/{i}.png")
    # segmentation: flat image dir + matching mask dir
    os.makedirs(f"{DATA}/seg/images", exist_ok=True)
    os.makedirs(f"{DATA}/seg/masks", exist_ok=True)
    for i in range(8):
        Image.fromarray(rng.integers(0, 255, (IMG, IMG, 3), dtype=np.uint8)).save(f"{DATA}/seg/images/{i:03d}.png")
        Image.fromarray(rng.integers(0, N_CLASSES, (IMG, IMG), dtype=np.uint8)).save(f"{DATA}/seg/masks/{i:03d}.png")

build_data()
rng = np.random.default_rng(1)
SEQ = (rng.normal(size=(40, 30, 1)).astype("float32"), rng.integers(0, N_CLASSES, 40))
TAB = (rng.normal(size=(60, 12)).astype("float32"), rng.integers(0, N_CLASSES, 60))
REC = (rng.random((60, 24)).astype("float32"),)
TOK = (rng.integers(0, 500, (40, 30)), rng.integers(0, N_CLASSES, 40))

def onehot(y):
    return np.eye(N_CLASSES)[np.asarray(y)]

# Which fixture each case needs. Anything not listed uses the image folders.
SEQUENCE = {"dl_lstm_scratch", "dl_gru_scratch", "custom_LSTM", "custom_GRU"}
TOKENS = {"dl_transformer_scratch"}  # modality: text
RECON = {"dl_autoencoder_scratch"}
SEGMENT = {"dl_unet_scratch"}
TABULAR = {"custom_Dense", "custom_Dropout", "custom_Flatten", "custom_BatchNorm"}

def prepare(name, src):
    """Point the script at data. No other edit is permitted."""
    if name.startswith("ml_"):
        ds = ("from sklearn.datasets import load_diabetes\n_d = load_diabetes(); X, y = _d.data, _d.target\n"
              if "linearregression" in name else
              "from sklearn.datasets import load_iris\n_d = load_iris(); X, y = _d.data, _d.target\n")
        return src.replace("\n# Guard:", "\n" + ds + "\n# Guard:", 1), {}
    if name in SEGMENT:
        return (src.replace('IMAGE_DIR = "path/to/images"', f'IMAGE_DIR = "{DATA}/seg/images"')
                   .replace('MASK_DIR = "path/to/masks"', f'MASK_DIR = "{DATA}/seg/masks"')), {}
    if name in SEQUENCE:
        return src.replace("# X = ...\n# y = ...", "X = _X\ny = _y"), {"_X": SEQ[0], "_y": onehot(SEQ[1])}
    if name in TOKENS:
        return src.replace("# X = ...\n# y = ...", "X = _X\ny = _y"), {"_X": TOK[0], "_y": onehot(TOK[1])}
    if name in RECON:
        return src.replace("# X = ...", "X = _X"), {"_X": REC[0]}
    if name in TABULAR:
        return src.replace("# X = ...\n# y = ...", "X = _X\ny = _y"), {"_X": TAB[0], "_y": onehot(TAB[1])}
    return src.replace('DATA_DIR = "path/to/your/image_dataset"', f'DATA_DIR = "{DATA}/images"'), {}

# --- run -------------------------------------------------------------------
rows = []
for path in sorted(glob.glob(f"{GEN}/*.py")):
    name = os.path.basename(path)[:-3]
    src = open(path).read()
    try:
        snippet, extra = prepare(name, src)
    except Exception as e:
        rows.append({"case": name, "status": "SKIP", "detail": f"no fixture: {e}"}); continue

    status, detail = "pass", ""
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(snippet, path, "exec"), {"__name__": "__main__", **extra})
    except Exception as e:
        msg = str(e).splitlines()[0]
        # Distinguish an environment limitation from a defect in the generated code.
        if "403" in msg or "URL fetch" in msg:
            status, detail = "BLOCKED", "ImageNet weight download unavailable in this environment"
        else:
            status, detail = "FAIL", f"{type(e).__name__}: {msg[:150]}"
    if status == "pass" and "# Unsupported" in src:
        status, detail = "FAIL", "no model emitted"
    rows.append({"case": name, "status": status, "detail": detail})
    print(f"{status:8} {name:32} {detail}")

json.dump(rows, open("./run_all_report.json", "w"), indent=2)
n = {k: sum(r["status"] == k for r in rows) for k in ("pass", "FAIL", "BLOCKED", "SKIP")}
print(f"\n{n['pass']} pass / {n['FAIL']} fail / {n['BLOCKED']} blocked / {n['SKIP']} skipped, {len(rows)} total")
