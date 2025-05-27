# Spider web analysis

## Installing

How to run the code:

- Install juliaup
- Run `juliaup add lts` to install Julia 1.10 LTS via juliaup
- Clone this repository locally
- Open it on VSCode with the Julia extension installed
- Run `] activate .` on the Julia console. You'll see the REPL change to `(TDAweb) pkg> `
- Run `] precompile` on the Julia console. It'll install all dependencies
- Try some code! For example, "using/preprocessing.jl" has some useful functions

How to edit the article:
- Read the [general instructions here](https://quarto.org/docs/manuscripts/authoring/vscode.html#project-files)
- Install the Quarto extension for VSCode
- open `article.qmd` and edit as you wish! It uses Markdown to format text. You can read more at [the Quarto website](https://quarto.org/docs/authoring/markdown-basics.html)

## Repository structure

The `images` directory contains a directory for each species, and this name is used in the analysis:

```
images 
- SPINOSAD
- ENDOSULFAN
...
```