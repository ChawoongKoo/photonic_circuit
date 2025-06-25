from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ automatically

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from models import coupler, waveguide, mzi

w1 = jnp.linspace(1.51, 1.59, 1000)
s_matrix = mzi(w1=w1, top={"length": 25.}, bot={"length": 15.})

plt.plot(w1*1e3, abs(s_matrix["in0", "out0"])**2)
plt.ylim(-.05, 1.05)
plt.xlabel("nm")
plt.ylabel("T")
plt.show()
