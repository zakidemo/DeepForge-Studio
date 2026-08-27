import ast, glob, io, os, contextlib, json, random
random.seed(0)
files = sorted(glob.glob('sweep/*.py'))
syn = []
for f in files:
    try: ast.parse(open(f).read())
    except SyntaxError as e: syn.append((f, str(e)))
print(f"syntax: {len(files)-len(syn)}/{len(files)} valid")
for f, e in syn[:10]: print("  ", os.path.basename(f), e)

# execute a random sample of each model's configurations
by_model = {}
for f in files: by_model.setdefault(os.path.basename(f).split('__')[0], []).append(f)
fails = []
for model, fs_ in by_model.items():
    sample = random.sample(fs_, min(25, len(fs_)))
    for f in sample:
        src = open(f).read()
        d = ("from sklearn.datasets import load_diabetes\n_d = load_diabetes(); X, y = _d.data, _d.target\n"
             if model == 'linearregression' else
             "from sklearn.datasets import load_iris\n_d = load_iris(); X, y = _d.data, _d.target\n")
        src = src.replace("\n# Guard:", "\n" + d + "\n# Guard:", 1)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(src, f, 'exec'), {'__name__': '__main__'})
        except Exception as e:
            fails.append((os.path.basename(f), f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"))
print(f"\nexecuted sample; {len(fails)} runtime failures")
for f, e in fails[:20]: print("  ", f, e)
