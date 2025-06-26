import sax
import jax
import jax.numpy as jnp

# NOTE 
# Create a component class later on to organize these models

# NOTE
# Sdicts are a datatype in SAX that are essentially dictionary objects.
# The keys are different combination of input/output ports and the values 
# are the corresponding transmission/reflection 

# definition of a simple coupler
# returns transmission/reflection of input/output port combinations 
def coupler(coupling=0.5) -> sax.SDict:
    kappa = coupling**0.5
    tau = (1-coupling)**0.5
    sdict = sax.reciprocal({
        ("in0", "out0"): tau,
        ("in0", "out1"): 1j*kappa,
        ("in1", "out0"): 1j*kappa,
        ("in1", "out1"): tau
    })

    return sdict

# definition of a simple waveguide
# returns the transmission through the waveguide
def waveguide(w1=1.55, w10=1.55, neff=2.34, ng=3.4, length=10, loss=0.) -> sax.SDict:
    dw1 = w1 - w10
    dneff_dw1 = (ng - neff) / w10
    neff = neff - (dw1 * dneff_dw1)
    phase = 2*jnp.pi *neff*length/w1
    amplitude = jnp.asarray(10**(-loss*length/20), dtype=complex)

    transmission = amplitude*jnp.exp(1j*phase)
    sdict = sax.reciprocal({("in0", "out0"): transmission})
    return sdict

# Definition of a Mach Zehnder Interferometer
# It is a composite component, basically being
# a circuit made up of couplers and waveguides
mzi, info = sax.circuit(
    netlist={
        "instances": {
            "left": "coupler",
            "top": "waveguide",
            "bot": "waveguide",
            "right": "coupler"
        },
        "connections": {
            "left,out0": "bot,in0",
            "bot,out0": "right,in0",
            "left,out1": "top,in0",
            "top,out0": "right,in1"
        },
        "ports": {
            "in0": "left,in0",
            "in1": "left,in1",
            "out0": "right,out0",
            "out1": "right,out1"
        }
    },
    models = {
        "coupler": coupler,
        "waveguide": waveguide
    }
)

# Directional coupler definition 
# for use in a chain of MZI's
# Essentially the same as the coupler component
dc_with_arms, info = sax.circuit(
    netlist={
        "instances": {
            "left": "coupler",
            "bot": "waveguide",
            "top": "waveguide"
        },
        "connections": {
            "left,out0": "bot,in0",
            "left,out1": "top,in0"
        },
        "ports": {
            "in0": "left,in0",
            "in1": "left,in1",
            "out0": "bot,out0",
            "out1": "top,out0"
        }
    },
    models={
        "coupler": coupler,
        "waveguide": waveguide
    }
)

# Function/model factory that returns a chain of 
# MZI's implemented by taking the output of a pervious directional coupler
# and linking it with the input of the next coupler
def mzi_chain(num_mzis=1) -> sax.Model:
    chain, _ = sax.circuit(
        netlist= {
            "instances": {f"dc{i}": "dc_with_arms" for i in range(num_mzis+1)},
            "connections": {
                **{f"dc{i},out0": f"dc{i+1},in0" for i in range(num_mzis)},
                **{f"dc{i},out1": f"dc{i+1},in1" for i in range(num_mzis)}
            },
            "ports": {
                "in0": "dc0,in0",
                "in1": "dc0,in1",
                "out0": f"dc{num_mzis},out0",
                "out1": f"dc{num_mzis},out1"
            }
        },
        models={"dc_with_arms": dc_with_arms},
        backend="klu",
        # Return type being SDict for sparse matrix
        # or SDense for dense matrix, among others
        return_type="SDense"
    )
    # Makes 'chain' return an sdict object when called
    return sax.sdict(chain)

