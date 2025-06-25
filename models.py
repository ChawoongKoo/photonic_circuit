import sax
import jax
import jax.numpy as jnp

# NOTE
# Sdicts are a datatype in SAX that are essentially dictionary objects.
# The keys are different combination of input/output ports and the values 
# are the corresponding transmission/reflection 

# definition of a simple coupler
# returns transmission/reflection of input/output port combinations 
def coupler(coupling=0.5):
    kappa = coupling**0.5
    tau = (1-coupling)**0.5
    sdict = sax.reciprocal({
        ("in0", "out0"): tau,
        ("in0", "out1"): 1j*kappa,
        ("in1", "out0"): 1j*kappa,
        ("in1", "out1"): tau
    })

    return sdict

# print(coupler(coupling=.3))

#definition of a simple waveguide
#returns the transmission through the waveguide
def waveguide(w1=1.55, w10=1.55, neff=2.34, ng=3.4, length=10, loss=0.):
    dw1 = w1 - w10
    dneff_dw1 = (ng - neff) / w10
    neff = neff - (dw1 * dneff_dw1)
    phase = 2*jnp.pi *neff*length/w1
    amplitude = jnp.asarray(10**(-loss*length/20), dtype=complex)

    transmission = amplitude*jnp.exp(1j*phase)
    sdict = sax.reciprocal({("in0", "out0"): transmission})
    return sdict

# print(waveguide(length=100))

