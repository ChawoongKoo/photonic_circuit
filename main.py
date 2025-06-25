from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ automatically

import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as opt
import sax
import matplotlib.pyplot as plt
from models import coupler, waveguide, mzi
import tqdm

# Basic simulation of an MZI's transmission for various wavelengths of input light
def basic_sim():
    w1 = jnp.linspace(1.51, 1.59, 1000)
    s_matrix = mzi(w1=w1, top={"length": 25.}, bot={"length": 15.})

    plt.plot(w1*1e3, abs(s_matrix["in0", "out0"])**2)
    plt.ylim(-.05, 1.05)
    plt.xlabel("nm")
    plt.ylabel("T")
    plt.show()

# basic_sim()

# Simulation of adjusting the length of the top waveguide in an MZI 
# so that it blocks transmission of 1550 nm light. Uses gradient descent.
def gradient_sim():
    # Loss function definition
    def loss_fn(delta_length) -> jax.Array:
        s_matrix = mzi(w1=1.55, top={"length": 15. + delta_length}, bot={"length": 15.})
        return (abs(s_matrix["in0", "out0"])**2)
    
    # Gradient function definition, argnums=0 means we optimize loss_fn's argument 0
    grad_fn = jax.jit(
        jax.grad(
            loss_fn,
            argnums=0
        )
    )

    # Initial function gives us initial state
    # Update function updates our state
    # Params function gives us parameters of a given state
    init_delta_length = 10.
    init_fn, update_fn, params_fn = opt.adam(step_size=.1)
    state = init_fn(init_delta_length)

    # Step function definition
    def step_fn(step, state):
        settings = params_fn(state)
        loss = loss_fn(settings)
        grad = grad_fn(settings)
        state = update_fn(step, grad, state)
        return loss, state

    range_ = tqdm.trange(300)
    for step in range_:
        loss, state = step_fn(step=step, state=state)
        range_.set_postfix(loss=f"{loss:.6f}")

    delta_length = params_fn(state)
    # print(delta_length)
    w1 = jnp.linspace(1.51, 1.59, 1000)
    s_matrix = mzi(w1=w1, top={"length": 15. + delta_length}, bot={"length": 15.})
    print(abs(s_matrix["in1", "out1"])**2)

    plt.plot(w1*1e3, abs(s_matrix["in1", "out1"])**2)
    plt.xlabel("nm")
    plt.ylabel("T")
    plt.title("Transmission from input 1 to output 1, minimized at 1550 nm light")

    plt.plot([1550,1550], [0,1])
    plt.show()

gradient_sim()