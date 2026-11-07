"""Load frozen material functions with explicit archived source inputs.

Only private module globals are redirected; historical source files and plans
are never rewritten. This restores the original renderer import without the
unavailable former machine's directory layout.
"""
import importlib.util
from pathlib import Path
import sys
from common import sha

def load_builder(base,extension):
    base,root=Path(base).resolve(),Path(extension).resolve()
    renderer_path=base/'data/e9_affine_balance/source_render_v2.py'
    spec=importlib.util.spec_from_file_location('render_v2',renderer_path)
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    previous=sys.modules.get('render_v2');sys.modules['render_v2']=renderer
    builder_path=base/'code/build_v3_data.py'
    try:
        spec=importlib.util.spec_from_file_location('isolated_frozen_v3_material_builder',builder_path)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    finally:
        if previous is None:sys.modules.pop('render_v2',None)
        else:sys.modules['render_v2']=previous
    builder.SOURCE=root/'data/argkp_source';builder.PROJECT=root
    return builder,{str(builder_path):sha(builder_path),str(renderer_path):sha(renderer_path),str(Path(__file__)):sha(__file__)}
