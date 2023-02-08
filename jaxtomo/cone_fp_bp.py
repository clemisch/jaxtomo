import jax

from . import cone_fp
from . import cone_bp


def get_fp_bp(angles, X, dX, Y, U, dU, V, dV, S, D):
    @jax.custom_vjp
    def project(vol):
        proj = cone_fp.get_fp(vol, angles, dX, U, dU, V, dV, S, D)
        return proj

    def _project_fwd(vol):
        return project(vol), None

    def _project_bwd(res, g):
      return (backproject(g),)

    project.defvjp(_project_fwd, _project_bwd)

    @jax.custom_vjp
    def backproject(proj):
        vol = cone_bp.get_bp(proj, angles, dU, dV, X, Y, dX, S, D)
        return vol

    def _backproject_fwd(proj):
        return backproject(proj), None

    def _backproject_bwd(res, g):
      return (project(g),)

    backproject.defvjp(_backproject_fwd, _backproject_bwd)

    project = jax.jit(project)
    backproject = jax.jit(backproject)

    return project, backproject
