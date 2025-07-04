from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ automatically

import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as opt
import sax
import matplotlib.pyplot as plt
from models import coupler, waveguide, mzi, mzi_chain
import tqdm

