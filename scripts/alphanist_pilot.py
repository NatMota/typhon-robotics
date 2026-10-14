"""Use existing environments without installing or modifying dependencies."""
import os,sys,site
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
site.addsitedir(os.environ.get("TYPHON_SIM_SITE",str(Path.home()/"Projects/humanoid-task-supervisor/.venv-sim/lib/python3.11/site-packages")))
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
from experiments.alphanist_pilot.run import main
if __name__=="__main__":main()
