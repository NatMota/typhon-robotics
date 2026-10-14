import mujoco,json,pathlib
p=pathlib.Path('/var/tmp/typhon-load05q-20261014/C/before/model.xml')
m=mujoco.MjModel.from_xml_path(str(p))
j={'version':mujoco.__version__,'solver':int(m.opt.solver),'solver_name':str(mujoco.mjtSolver(m.opt.solver)),'integrator':int(m.opt.integrator),'jacobian':int(m.opt.jacobian),'isDual':int(mujoco.mj_isDual(m)),'isSparse':int(mujoco.mj_isSparse(m)),'enableflags':int(m.opt.enableflags),'disableflags':int(m.opt.disableflags),'timestep':m.opt.timestep,'nv':m.nv,'nq':m.nq,'nu':m.nu,'ntendon':m.ntendon,'nplugin':m.nplugin}
print(json.dumps(j))
pathlib.Path('/var/tmp/typhon-load05r-20261014/model-options.json').write_text(json.dumps(j,indent=2))
