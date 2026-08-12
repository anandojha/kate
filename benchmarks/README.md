# benchmarks

- `benchmark_traj.py` : one command kinetic fidelity benchmark, all tools, one frozen ruler per complex

## Environment (one env, all tools)

```
conda create -n benchmark_traj python=3.11 -y
conda activate benchmark_traj
conda install -c conda-forge mdtraj deeptime matplotlib cmake cxx-compiler c-compiler make git -y
pip install torch zfpy fpzip
pip install "kate[kinetics,test] @ git+https://github.com/anandojha/kate.git"
```

- CLI tools (SZ3, SPERR, MDCompress): build per [`../COMPRESSORS.md`](../COMPRESSORS.md), install into the env prefix (`-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX`)
- SPERR runtime: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64` (shared object lands in lib64)
- neural baselines (optional): compressTraj and MDZip per [`../COMPRESSORS.md`](../COMPRESSORS.md); point `KATE_CT_DIR` / `KATE_MDZIP_MODEL` at them
- pcazip (optional): bioconda `pcasuite`, separate env if the solver conflicts

## Data

- DESRES fast folder trajectories (Lindorff-Larsen et al., Science 334, 517, 2011)
- available from D. E. Shaw Research on request; not redistributable here
- expected layout per complex, alpha carbon variant:

```
<name>/
  run0-ca/<DESRES inner directory>/*.dcd
  run1-ca/...
```

- DCD chunks per run: sorted, concatenated; frame interval 0.2 ns

## Run

```
python benchmark_traj.py NAMEDIR STRIDE LAG NSTATES NAME \
  --tools sz3,zfp,fpzip,sperr,xtc,pca,kate \
  --sz3-bin $CONDA_PREFIX/bin/sz3 --sperr-bin $CONDA_PREFIX/bin/sperr3d
```

- tools: sz3, zfp, fpzip, sperr, xtc, pca, mdc, ct, mdzip, kate
- missing tool: SKIPPED row, run continues
- env knobs: KATE_SZ3_BIN, KATE_SPERR_BIN, KATE_MDCOMPRESS_BIN, KATE_CT_DIR, KATE_CT_PYTHON, KATE_MDZIP_MODEL, KATE_MDZIP_EPOCHS

## Per complex parameters (the published twelve)

| name | stride | lag | nstates |
|---|---|---|---|
| chignolin | 1 | 500 | 150 |
| trp_cage | 2 | 1000 | 200 |
| bba | 6 | 900 | 150 |
| villin | 4 | 210 | 150 |
| ww_domain | 20 | 315 | 200 |
| ntl9 | 28 | 621 | 200 |
| bbl | 12 | 1450 | 200 |
| protein_b | 3 | 780 | 150 |
| homeodomain | 3 | 620 | 150 |
| protein_g | 24 | 3385 | 200 |
| alpha3d | 9 | 900 | 200 |
| lambda_repressor | 12 | 1225 | 200 |

- lag in strided frames, chosen by implied timescale scan against the published folding time
- runs pooled per complex; counts never span a run boundary

## SLURM

```
sbatch -J bt_NAME --export=ALL,NAME=...,STRIDE=...,LAG=...,NSTATES=...,TOOLS=sz3:zfp:fpzip:sperr:xtc:pca:kate job.sbatch
```

- `--export` splits on commas: pass TOOLS colon separated, expand with `${TOOLS//:/,}` inside the job script

## Produces

- `NAME_benchmark.csv` : tool, param, bits_per_coord, rmsd_A, t1_ns, folding_err_pct
- one frozen reference discretization per complex; every reconstruction scored through it
- `kate_stored` : stored MSM, kinetics only rate
- `kate_roundtrip` : full length reconstruction (`kate.runner.reconstruct_full_length`), re-estimated MSM, full rate
- rmsd_A : per frame optimal superposition; lower than raw frame differences by construction
