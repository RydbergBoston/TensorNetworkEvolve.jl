# Cracker

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Roger-luo.github.io/Cracker.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Roger-luo.github.io/Cracker.jl/dev)
[![Build Status](https://github.com/Roger-luo/Cracker.jl/workflows/CI/badge.svg)](https://github.com/Roger-luo/Cracker.jl/actions)
[![Coverage](https://codecov.io/gh/Roger-luo/Cracker.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Roger-luo/Cracker.jl)

[Tracker](https://github.com/FluxML/Tracker.jl) but for [ChainRules](https://github.com/JuliaDiff/ChainRules.jl). Based on `Tracker` and `ReverseDiff` but built with `ChainRules` in mind.

## Installation

<p>
Cracker is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install Cracker,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd> key in the REPL to use the package mode, then type the following command
</p>

```julia
pkg> add Cracker
```

## Why Operator Overloading?

Before good source-to-source AD engine becomes mature, an operator overloading
AD engine that always builds a linear trace would be nice to have. These
not so fancy implementation is quite useful when the program doesn't require
control-flow to analyze and is fast to compile because of this simplicity.

## Why not ...?

There has been several existing implementations, however

- [Tracker](https://github.com/FluxML/Tracker.jl) is under maintainance mode, it won't support ChainRules ecosystem, and it won't support [complex number](https://github.com/FluxML/Tracker.jl/pull/16)
- [Nabla](https://github.com/invenia/Nabla.jl) aims for sensitivity analysis and scalar case instead high performance array operations, e.g one will need to change the element type to `Leaf`. which may prevent one from using it on GPUs.
- [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl) has similar problem as `Tracker`

to summary, we need an operator-overloading AD engine for the era of `ChainRules` that has
good array operation support.

## Why Tracker style?

the old `Tracker` has some good design choice for simple programs, the explicit tracked-types
enable one to use the AD engine on simple programs intuitively.

## License

MIT License
