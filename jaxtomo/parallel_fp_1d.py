import jax
import jax.numpy as jnp
from functools import partial


def get_ray(img, theta, y, xx):
    def get_point(s, img_slice):
        x = 1. / jnp.cos(theta) * (y - s * jnp.sin(theta))
        x_interp = jnp.interp(x, xx, img_slice, left=0., right=0.)  
        return x_interp

    points = jax.vmap(get_point)(xx, img.T)
    ray = jnp.sum(points)
    ray = ray / jnp.cos(theta)

    return ray 


@partial(jax.jit, static_argnames=("I", "P"))
def get_proj(img, theta, sigma, I, delta, P):

    def cond_fun(arg):
        angle, _ = arg
        is_valid_angle = jnp.logical_and(
            (-jnp.pi / 4) < angle, 
            angle <= (jnp.pi / 4)
        )
        return jnp.logical_not(is_valid_angle)

    def body_fun(arg):
        angle, im = arg
        im = im.T
        im = im[:, ::-1]
        angle = angle - jnp.pi / 2
        return angle, im

    theta, img = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (theta, img)
    )

    # width in pixels => one voxel/pixel smaller than physical image/detector!
    width_img = sigma * (I - 1)
    width_proj = delta * (P - 1)

    # origins
    O_I = sigma * (-0.5 * I + 0.5)
    O_P = delta * (-0.5 * P + 0.5)

    xx = jnp.linspace(0., 1., I, endpoint=True) * width_img + O_I
    yy = jnp.linspace(0., 1., P, endpoint=True) * width_proj + O_P

    proj = jax.vmap(get_ray, (None, None, 0, None), 0)(img, theta, yy, xx)

    return proj


@partial(jax.jit, static_argnames=("I", "P"))
def get_projs_vmap(img, thetas, sigma, I, delta, P):
    projs = jax.vmap(get_proj, (None, 0, None, None, None, None), 0)(
        img, thetas, sigma, I, delta, P
    )
    return projs


@partial(jax.jit, static_argnames=("I", "P"))
def get_projs_map(img, thetas, sigma, I, delta, P):
    projs = jax.lax.map(
        lambda theta: get_proj(img, theta, sigma, I, delta, P),
        thetas
    )
    return projs
