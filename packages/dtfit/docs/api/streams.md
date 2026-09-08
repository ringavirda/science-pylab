# Streams and scale

The image accumulates in fixed memory: a running `ImageStream` folds each
chunk of samples into the same `(S, G)` statistic the batch fitters build in
one pass. Blocks assemble onto a coarser domain through an exact transfer,
channels batch through one shared GEMM, and independent fits fan out across
processes with `fit_many`.

::: dtfit.ImageStream

::: dtfit.image.assemble

::: dtfit.image.legendre_transfer

::: dtfit.image.block_transfer

::: dtfit.fit_many

::: dtfit.image.FittingProblem
