# FilterBank & FusedChiSquareDetector -- experimental tooling

`FilterBank` (many streaming filters driven in lockstep) and
`FusedChiSquareDetector` (a pooled multi-stream fault test built on them)
live in `dtfit_experimental.streaming` -- experiment harness tooling, not
part of the stable `dtfit` surface. See
[the experimental adaptations API](Experimental-Adaptations-API).

The stable equivalent of the fused test is summing several
[`ImageFilter`](Methods-Legendre-Filter) instances' `nis_`: each is
chi-square under its model, so the sum is chi-square with the summed
degrees of freedom, and a fault that moves every stream is weak in any
one filter's innovation but strong in the pooled statistic. See
[several streams](API-Streaming#several-streams).
