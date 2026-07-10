# Installing the baseline compressors

KATE is benchmarked against six external compressors. Each is invoked by
`kate/baselines.py` as a subprocess or through its Python bindings, so none of
them is a build dependency of KATE itself. The commands below reproduce the
installation used for the twelve-protein benchmark. Versions in parentheses are
the ones tested.

All error-bounded coordinate compressors read and write raw little-endian
`float32` arrays. A trajectory of `T` frames and `N` atoms is written as a flat
array of `3*N*T` floats, and the decompressed array is reshaped back to
`(T, N, 3)`.

## SZ3 (error-bounded, prediction-based; built from source)

```
git clone https://github.com/szcompressor/SZ3.git
cd SZ3 && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$PWD/install" ..
make -j && make install
```

The CLI is `build/install/bin/sz3`. Absolute-error compression and
decompression of a length-`N` float32 array:

```
sz3 -f -z out.sz  -i in.f32  -M ABS <eb> -1 <N>     # compress at absolute error <eb>
sz3 -f -x out.f32 -s out.sz              -1 <N>     # decompress
```

## ZFP (fixed-rate / fixed-accuracy transform coding)

```
pip install zfpy                                    # (zfpy 1.0.1)
```

KATE drives the fixed-rate mode through the Python bindings. The C library and
its `zfp` CLI can instead be built with

```
git clone https://github.com/LLNL/zfp.git
cd zfp && mkdir build && cd build && cmake .. && make -j
```

## fpzip (near-lossless floating-point; Python bindings)

```
pip install fpzip                                   # (fpzip 1.2.5)
```

The precision is set by the number of retained mantissa bits, swept to trace the
rate axis.

## SPERR (wavelet plus SPECK; built from source)

```
git clone https://github.com/NCAR/SPERR.git
cd SPERR && mkdir build && cd build
cmake -DBUILD_CLI_UTILITIES=ON -DUSE_OMP=ON -DCMAKE_INSTALL_PREFIX="$PWD/install" ..
make -j && make install
```

The 3D CLI is `sperr3d`. It must find `libSPERR.so`, so run it with

```
LD_LIBRARY_PATH="<install>/lib64" \
  sperr3d --pwe <err> --dims 3 <N> <T> -o out.bin in.f32
```

The coordinate array is presented to `sperr3d` as a 3D volume of shape
`(3, N, T)` so that the wavelet transform runs along atoms and time, and `--pwe`
sets the point-wise error bound.

## MDZip (neural convolutional autoencoder; De Silva and Perez 2025)

```
git clone https://github.com/PDNALab/MDZip.git
cd MDZip && pip install -e .
```

MDZip trains one autoencoder per system and stores the latent codes, so it needs
`torch`, `numpy`, and a trajectory reader (`mdtraj` or `MDAnalysis`). The
benchmark uses the residual skip-connection variant (`skipAE`) at bottleneck
`z = 20`. The public repository ships the method and an alanine-dipeptide demo
but none of the paper's benchmark trajectories.

## MGARD (optional; multigrid error-bounded)

```
git clone https://github.com/CODARcode/MGARD.git
cd MGARD && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$PWD/install" .. && make -j && make install
```

Not part of the twelve-protein sweep; included here for completeness as an
alternative error-bounded coordinate compressor.

## GROMACS XTC and pcazip (reference points)

XTC fixed-precision quantization is available through GROMACS
(`gmx trjconv -o traj.xtc`), and principal-component projection through `pcazip`
or the in-harness PCA in `kate/baselines.py`. Both are reported as coordinate-
fidelity reference points rather than swept across a rate ladder.
