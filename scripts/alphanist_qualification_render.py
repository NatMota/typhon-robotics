import os, sys, site
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
site.addsitedir(str(Path.home()/"Projects/humanoid-task-supervisor/.venv-sim/lib/python3.11/site-packages"))
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
from experiments.alphanist_qualification.render import main
if __name__=="__main__":main()
