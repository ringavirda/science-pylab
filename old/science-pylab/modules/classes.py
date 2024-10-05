from typing import Callable
import numpy as np
import sympy as sp
from sympy.abc import k, n, w

from modules.discretes import Discretes


class Model:
    def __init__(self, raw_expr: str, main_var: str) -> None:
        self.__raw: str = raw_expr
        self.__f: Callable[[np.float64], np.float64] = None
        self.sym: sp.Expr = sp.parse_expr(raw_expr)
        self.coeffs: list[sp.Symbol] = sorted(
            self.sym.free_symbols, key=lambda s: s.name
        )
        self.var: sp.Symbol = sp.var(main_var)
        self.coeffs.remove(self.var)
        self.coeffs_trained: np.ndarray = []
        self.rank: int = len(self.coeffs)

    def __str__(self) -> str:
        return self.__raw

    def as_lambda(self) -> Callable[[np.float64], np.float64]:
        if self.__f is None:
            raise RuntimeError("Unable to lambify untrained.")

        return self.__f

    def spectrum(self, rank=0, offset=False) -> list[sp.Expr]:
        reflection = self.__reflect(self.sym)
        spectr = []

        rank = len(self.coeffs) if rank == 0 else rank
        for i in range(rank):
            spectr.append(reflection.subs(k, i))

        if offset:
            model_plain = self.sym.copy()
            for a in self.coeffs:
                model_plain = model_plain.subs(a, 1)
            series_plain = sp.series(model_plain, self.var, n=rank).removeO()
            coeffs_mp = sp.poly(series_plain, self.var).coeffs()
            # Get the spectrum and balance it against poly coeffs
            for i, d in enumerate(spectr):
                spectr[i] = d / coeffs_mp[i]

        return spectr

    def __reflect(self, sequence: sp.Expr, add: bool = True) -> sp.Expr:
        reflection = None
        for term in sequence.args:
            discrete = term
            if term.is_Symbol:
                if term == self.var:
                    discrete = Discretes.C.subs(((w, 1), (n, 1)))
                elif add:
                    discrete = Discretes.I.subs(w, term)
            elif term.is_Pow:
                discrete = Discretes.C.subs(n, term.args[1])
            elif term.is_Mul:
                discrete = self.__reflect(term, add=False)
            elif isinstance(term, sp.exp):
                discrete = (
                    Discretes.E.subs(w, term.args[0].args[0])
                    if term.args[0].is_Mul
                    else Discretes.E.subs(w, 1)
                )
            elif isinstance(term, sp.sin):
                discrete = (
                    Discretes.Sin.subs(w, term.args[0].args[0])
                    if term.args[0].is_Mul
                    else Discretes.Sin.subs(w, 1)
                )
            elif isinstance(term, sp.cos):
                discrete = (
                    Discretes.Cos.subs(w, term.args[0].args[0])
                    if term.args[0].is_Mul
                    else Discretes.Cos.subs(w, 1)
                )

            if not reflection:
                reflection = discrete
            elif add:
                reflection += discrete
            else:
                reflection *= discrete

        return reflection

    def apply_trained(self, coeffs: np.ndarray) -> None:
        if len(coeffs) != len(self.coeffs):
            raise RuntimeError("Invalid amount of trained coeffs.")

        model: sp.Expr = self.sym
        for i, a in enumerate(self.coeffs):
            model = model.subs(a, coeffs[i])
        self.__f = sp.lambdify(self.var, model)
        self.coeffs_trained = coeffs

    def collapse(self, value: np.float64, length: np.float64 = 0) -> np.float64:
        if self.__f is None:
            raise RuntimeError("Unable to collapse an untrained model.")
        if length == 0:
            return self.__f(value)
        else:
            collapsed = np.arange(value, value + length, dtype=np.float64)
            with np.nditer(collapsed, op_flags=["readwrite"]) as it:
                for value in it:
                    value[...] = self.collapse(value)
            return collapsed


class Polynomial(Model):
    def __init__(self, rank: int, main_var: str = "t") -> None:
        super().__init__(self.__construct_expr(rank, main_var), main_var)

    def __construct_expr(self, rank: int, main_var: str) -> str:
        terms: list[str] = ["c0"]
        for i in range(1, rank):
            terms.append(f"c{i}*{main_var}**{i}")
        return " + ".join(terms)
