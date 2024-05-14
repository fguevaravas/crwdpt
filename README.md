# Cloaking for random walks using a discrete potential theory
This code reproduces the figures in the manuscript _Cloaking for random walks using a discrete potential theory_ by Trent DeGiovanni and Fernando Guevara Vasquez. 

For the associated preprint see: [arXiv:2405.07961](https://arxiv.org/abs/2405.07961).

## Instructions
Most of the code in this repository is written in [Julia]([https://julialang.org/), either as standalone scripts (`.jl` extension) or Jupyter notebooks (`.ipynb` extension). The file `Project.toml` contains the package dependencies that are not part of the Julia distribution, as of version 1.10.
* The figures for the paper can be reproduced using the Jupyter notebook `RW_cloaking_demo.ipynb`. The file `RWC.jl` needs to be in the same directory as `RW_cloaking_demo.ipynb`.
* The discrete potential theory identities are validated numerically on a random graph in the Jupyter notebook `validation.ipynb`.
* The animations for random walks can be reproduced using `rwanim.jl`. This is the only standalone script. To run it, change directory to the location of this repository and activate the environment (see these [instructions](https://pkgdocs.julialang.org/v1/environments/) or type `]activate .` in the Julia REPL). Note that this process can take a few minutes and generates a few thousand png files in the directory `frames`.
```
include("rwanim.jl")
rwanim.do_animation_and_video()
```

## Funding
This project was partially funded by the National Science Foundation Grants DMS-2008610 and DMS-2136198.

## License
Unless noted otherwise, the code is released under the following license (see also `LICENSE` file)
```
BSD 3-Clause License

Copyright (c) 2022, Trent DeGiovanni and Fernando Guevara Vasquez
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
