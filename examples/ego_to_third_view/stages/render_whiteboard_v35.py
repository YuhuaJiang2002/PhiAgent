"""Exact fixed-scene regression recipe; use batch.py for scene-independent contracts."""
from runtime import require_launcher
require_launcher()
from pathlib import Path
import runpy
import sys

recipe=Path(__file__).resolve().parents[1]/'recipes/whiteboard_v35'
sys.path.insert(0,str(recipe))
sys.argv=[str(recipe/'render_lab_demo.py'),'--clock-upright','--whiteboards',
          '--long-edge-arms','--stabilized-contact','--fixed-torso']
runpy.run_path(str(recipe/'render_lab_demo.py'),run_name='__main__')
