"""Record the current environment without changing the tested-version reference."""
from pathlib import Path
import importlib.metadata as metadata
import subprocess
import sys
from gs_search.artifacts import diagnostics, write_json


def main():
    output=Path('artifacts').resolve();output.mkdir(exist_ok=True,parents=True)
    write_json(output/'environment.json',diagnostics())
    (output/'environment-freeze.txt').write_text(subprocess.check_output(
        [sys.executable,'-m','pip','freeze'],text=True))
    direct=['torch','torchvision','gsplat','numpy','scipy','numba','plyfile','transformers',
            'huggingface-hub','Pillow','opencv-python-headless','safetensors','accelerate',
            'SAM-2','plotly','ninja','pytest','aiohttp','clarabel','cvxpy','cvxopt',
            'dijkstra3d','polytope','unfoldNd','open3d','sympy']
    installed=[]
    for name in direct:
        try: installed.append(f'{name}=={metadata.version(name)}')
        except metadata.PackageNotFoundError: pass
    (output/'requirements-installed.txt').write_text('\n'.join(installed)+'\n')
    print(output)


if __name__=='__main__': main()
