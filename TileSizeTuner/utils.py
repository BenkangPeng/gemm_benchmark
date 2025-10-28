def dump(content: str, path: str, fname: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, fname), "w") as f:
        f.write(content)