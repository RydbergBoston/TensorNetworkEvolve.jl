using Cracker
using Documenter

DocMeta.setdocmeta!(Cracker, :DocTestSetup, :(using Cracker); recursive=true)

makedocs(;
    modules=[Cracker],
    authors="Roger-Luo <rogerluo.rl18@gmail.com> and contributors",
    repo="https://github.com/Roger-luo/Cracker.jl/blob/{commit}{path}#{line}",
    sitename="Cracker.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Roger-luo.github.io/Cracker.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Roger-luo/Cracker.jl",
)
