# MPC Grinding Circuit

Model Predictive Control for a ball-mill grinding circuit with linear state-space models and augmented Kalman filter state estimation.

Based on Chen et al. (2007): "Application of model predictive control in ball mill grinding circuit",
Minerals Engineering 20(11):1099-1108.

## System Description

**Controlled outputs (CVs):**
- Sp: Particle size (%-200mesh)
- Dm: Mill solids concentration (% solids)
- Fc: Circulating load (t/h)
- Ls: Sump level (m)

**Manipulated inputs (MVs):**
- Ff: Fresh ore feed (t/h)
- Fm: Mill water flow (m³/h)
- Fd: Dilution water flow (m³/h)
- Vp: Pump speed (Hz)

## Running

```bash
cd examples/MPC_Grinding_Circuit
jupyter notebook MPC_Grinding_Circuit_MIMO.ipynb
```

## Linear Solver Configuration

IPOPT handles the optimization. The default linear solver is **MUMPS**, bundled with IPOPT, no extra setup needed.

### Solver Comparison

| Solver | Speed   | Memory | Best For                 |
|--------|---------|--------|--------------------------|
| MUMPS  | Good    | Higher | Default, no setup needed |
| MA27   | Fast    | Low    | Small-medium problems    |
| MA57   | Fast    | Medium | General purpose          |
| MA97   | Fastest | Higher | Large problems           |

---

## Optional: HSL Solvers

HSL linear solvers (MA27, MA57, MA97) are 2-5x faster than MUMPS on larger problems.

### HSL Installation (Linux, no conda)

A conda-free Linux setup that works with CasADi's IPOPT.

#### 1. (Optional) Create a Python virtual environment

```bash
python3 -m venv ~/venvs/mpc
source ~/venvs/mpc/bin/activate
pip install --upgrade pip
pip install neuralmpcx
```

#### 2. Install build tools and dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential gfortran pkg-config git wget tar
sudo apt install -y libopenblas-dev liblapack-dev libmetis-dev
```

#### 3. Get the Coin-HSL source from STFC

1. Download Coin-HSL for IPOPT from [HSL for IPOPT](https://www.hsl.rl.ac.uk/ipopt/) (free for academic use). You'll get a file like `coinhsl-YYYY.MM.DD.tar.gz`.

2. Put the tarball in a convenient folder, e.g. `~/Downloads`.

#### 4. Build Coin-HSL

```bash
# Clone the ThirdParty-HSL build scripts
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
cd ThirdParty-HSL

# Unpack the tarball you got from STFC inside ThirdParty-HSL folder
tar -xzf ~/Downloads/coinhsl-*.tar.gz

# ThirdParty-HSL expects the folder to be named exactly "coinhsl"
mv coinhsl-* coinhsl

# Configure & build with a user prefix
./configure --prefix="$HOME/coinhsl"
make -j"$(nproc)"
make install
```

You should now have:
```
$HOME/coinhsl/lib/libcoinhsl.so
```

#### 5. Configure IPOPT to use HSL

In the notebook, uncomment the HSL configuration in `LinearMpc.__init__()`:

```python
# Uncomment below to use HSL solvers (faster for large problems)
ipopt_opts["linear_solver"] = "ma27"  # or "ma57", "ma97"
ipopt_opts["hsllib"] = os.path.expanduser("~/coinhsl/lib/libcoinhsl.so")
```

IPOPT loads the HSL library from that path.
