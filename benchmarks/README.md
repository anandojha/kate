# benchmarks

- `benchmark_traj.py` : one-command kinetic-fidelity benchmark on a DESRES complex

## Run

```
python benchmark_traj.py NAMEDIR STRIDE LAG NSTATES NAME \
  --tools sz3,zfp,fpzip,sperr,xtc,pca,kate \
  --sz3-bin PATH --sperr-bin PATH --mdcompress-bin PATH
```

- NAMEDIR : directory of independent runs (`run*-ca`, DCD chunks inside)
- tools : sz3, zfp, fpzip, sperr, xtc, pca, mdc, ct, mdzip, kate
- missing tool -> SKIPPED row, run continues
- env knobs : KATE_SZ3_BIN, KATE_SPERR_BIN, KATE_MDCOMPRESS_BIN, KATE_CT_DIR, KATE_CT_PYTHON, KATE_MDZIP_MODEL, KATE_MDZIP_EPOCHS

## Produces

- `NAME_benchmark.csv` : tool, param, bits_per_coord, rmsd_A, t1_ns, folding_err_pct
- one frozen reference discretization per complex; every tool scored through it
- kate rows: `kate_stored` (stored MSM, kinetics-only rate) and `kate_roundtrip` (full-length reconstruction re-estimated, full rate)
