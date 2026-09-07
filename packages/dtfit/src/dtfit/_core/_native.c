/* dtfit._core._native: compiled numeric kernels for the hot paths of the
 * differential-transformation fitting methods.
 *
 * Model expressions are arbitrary, SymPy-lambdified at the Python level, so
 * model evaluation stays in NumPy. What lives here are the pure-numeric inner
 * loops the methods repeat thousands of times and that carry heavy per-call
 * Python / scipy overhead:
 *
 *   simpson_windows(y, x, starts, stops)        -> areas               (EAC, EAF)
 *   simpson_windows_rows(Y, x, starts, stops)   -> areas per row       (Jacobians)
 *   legendre_project(fv, qw, legvander, norm)   -> spectral coeffs     (LSI)
 *
 * The Simpson kernel reproduces scipy.integrate.simpson exactly: composite
 * Simpson on non-uniform samples, with the Cartwright last-interval correction
 * for an even number of points. Swapping it in does not change results.
 *
 * Build with build_native.py (clang). Pure-Python fallbacks live in
 * dtfit/_core/_kernels.py, so the package works whether or not this is
 * compiled.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* Composite Simpson on a single 1-D slice y[0..n), sampled at x[0..n).
 * Mirrors scipy.integrate.simpson (1-D, x given):
 *   - n <= 1            -> 0
 *   - n == 2            -> trapezoid of the single interval
 *   - n odd  (even #intervals) -> plain composite Simpson over all pairs
 *   - n even (odd  #intervals) -> composite Simpson over the first n-1 points
 *                                 plus the Cartwright correction for the last.
 */

/* Sum of non-uniform Simpson pairs for i = start, start+2, ... while i < stop,
 * each pair spanning indices (i, i+1, i+2). Matches scipy._basic_simpson with
 * the divide-by-zero guards (a zero denominator contributes 0). */
static double basic_simpson(const double *y, const double *x,
                            npy_intp start, npy_intp stop)
{
    double result = 0.0;
    for (npy_intp i = start; i < stop; i += 2) {
        double h0 = x[i + 1] - x[i];
        double h1 = x[i + 2] - x[i + 1];
        double hsum = h0 + h1;
        double hprod = h0 * h1;
        double h0divh1 = (h1 != 0.0) ? (h0 / h1) : 0.0;
        double inv = (h0divh1 != 0.0) ? (1.0 / h0divh1) : 0.0;
        double mid = (hprod != 0.0) ? (hsum * (hsum / hprod)) : 0.0;
        result += (hsum / 6.0) * (y[i] * (2.0 - inv)
                                  + y[i + 1] * mid
                                  + y[i + 2] * (2.0 - h0divh1));
    }
    return result;
}

static double simpson_1d(const double *y, const double *x, npy_intp n)
{
    if (n <= 1) {
        return 0.0;
    }
    if (n == 2) {
        double h = x[1] - x[0];
        return 0.5 * h * (y[0] + y[1]);
    }
    if (n % 2 == 1) {
        /* odd count -> even number of intervals: full composite Simpson. */
        return basic_simpson(y, x, 0, n - 2);
    }
    /* even count -> odd number of intervals: Simpson on the first n-1 points
     * plus the Cartwright correction for the trailing interval. */
    double result = basic_simpson(y, x, 0, n - 3);
    double h0 = x[n - 2] - x[n - 3];
    double h1 = x[n - 1] - x[n - 2];

    double den = 6.0 * (h1 + h0);
    double alpha = (den != 0.0) ? (2.0 * h1 * h1 + 3.0 * h0 * h1) / den : 0.0;
    den = 6.0 * h0;
    double beta = (den != 0.0) ? (h1 * h1 + 3.0 * h0 * h1) / den : 0.0;
    den = 6.0 * h0 * (h0 + h1);
    double eta = (den != 0.0) ? (h1 * h1 * h1) / den : 0.0;

    result += alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3];
    return result;
}

/* Borrow a contiguous C-double view of obj; sets a Python error and returns
 * NULL on failure. The returned array must be Py_DECREF'd by the caller. */
static PyArrayObject *as_f64(PyObject *obj, int ndim)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROMANY(
        obj, NPY_DOUBLE, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    return a;  /* PyArray_FROMANY already sets the error on failure */
}

static PyArrayObject *as_intp(PyObject *obj)
{
    return (PyArrayObject *)PyArray_FROMANY(
        obj, NPY_INTP, 1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
}

/* simpson_windows(y, x, starts, stops) -> areas[m]
 *
 * Integrates y over m windows, window k covering the half-open index span
 * [starts[k], stops[k]) of the shared (x, y) samples.
 */
static PyObject *py_simpson_windows(PyObject *self, PyObject *args)
{
    PyObject *yo, *xo, *so, *eo;
    if (!PyArg_ParseTuple(args, "OOOO", &yo, &xo, &so, &eo)) {
        return NULL;
    }

    PyArrayObject *y = as_f64(yo, 1);
    PyArrayObject *x = as_f64(xo, 1);
    PyArrayObject *starts = as_intp(so);
    PyArrayObject *stops = as_intp(eo);
    PyObject *out = NULL;
    if (!y || !x || !starts || !stops) {
        goto done;
    }

    npy_intp nx = PyArray_DIM(x, 0);
    if (PyArray_DIM(y, 0) != nx) {
        PyErr_SetString(PyExc_ValueError, "y and x must have equal length");
        goto done;
    }
    npy_intp m = PyArray_DIM(starts, 0);
    if (PyArray_DIM(stops, 0) != m) {
        PyErr_SetString(PyExc_ValueError, "starts and stops must match in length");
        goto done;
    }

    npy_intp dims[1] = {m};
    PyArrayObject *areas = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!areas) {
        goto done;
    }

    const double *yp = (const double *)PyArray_DATA(y);
    const double *xp = (const double *)PyArray_DATA(x);
    const npy_intp *sp = (const npy_intp *)PyArray_DATA(starts);
    const npy_intp *ep = (const npy_intp *)PyArray_DATA(stops);
    double *ap = (double *)PyArray_DATA(areas);

    /* Validate every window while the GIL is held (PyErr_* needs it). */
    for (npy_intp k = 0; k < m; ++k) {
        npy_intp a = sp[k], b = ep[k];
        if (a < 0 || b > nx || b < a) {
            Py_DECREF(areas);
            PyErr_Format(PyExc_IndexError,
                         "window %zd span [%zd, %zd) out of bounds [0, %zd)",
                         (Py_ssize_t)k, (Py_ssize_t)a, (Py_ssize_t)b,
                         (Py_ssize_t)nx);
            goto done;
        }
    }
    /* The integral loop is pure C over borrowed buffers and touches no Python
     * object, so the GIL can be dropped. A thread pool can then run many such
     * kernels concurrently (see dtfit.streaming._bank). */
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp k = 0; k < m; ++k) {
        ap[k] = simpson_1d(yp + sp[k], xp + sp[k], ep[k] - sp[k]);
    }
    Py_END_ALLOW_THREADS
    out = (PyObject *)areas;

done:
    Py_XDECREF(y);
    Py_XDECREF(x);
    Py_XDECREF(starts);
    Py_XDECREF(stops);
    return out;
}

/* simpson_windows_rows(Y, x, starts, stops) -> areas[nrows, m]
 *
 * Like simpson_windows but Y is 2-D (nrows, nx): each row is integrated over
 * the same m windows. Used to integrate a stack of parameter sensitivities
 * (Jacobian rows) in one call.
 */
static PyObject *py_simpson_windows_rows(PyObject *self, PyObject *args)
{
    PyObject *Yo, *xo, *so, *eo;
    if (!PyArg_ParseTuple(args, "OOOO", &Yo, &xo, &so, &eo)) {
        return NULL;
    }

    PyArrayObject *Y = as_f64(Yo, 2);
    PyArrayObject *x = as_f64(xo, 1);
    PyArrayObject *starts = as_intp(so);
    PyArrayObject *stops = as_intp(eo);
    PyObject *out = NULL;
    if (!Y || !x || !starts || !stops) {
        goto done;
    }

    npy_intp nrows = PyArray_DIM(Y, 0);
    npy_intp nx = PyArray_DIM(x, 0);
    if (PyArray_DIM(Y, 1) != nx) {
        PyErr_SetString(PyExc_ValueError,
                        "Y must have shape (nrows, len(x))");
        goto done;
    }
    npy_intp m = PyArray_DIM(starts, 0);
    if (PyArray_DIM(stops, 0) != m) {
        PyErr_SetString(PyExc_ValueError, "starts and stops must match in length");
        goto done;
    }

    npy_intp dims[2] = {nrows, m};
    PyArrayObject *areas = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!areas) {
        goto done;
    }

    const double *Yp = (const double *)PyArray_DATA(Y);
    const double *xp = (const double *)PyArray_DATA(x);
    const npy_intp *sp = (const npy_intp *)PyArray_DATA(starts);
    const npy_intp *ep = (const npy_intp *)PyArray_DATA(stops);
    double *ap = (double *)PyArray_DATA(areas);

    for (npy_intp k = 0; k < m; ++k) {
        npy_intp a = sp[k], b = ep[k];
        if (a < 0 || b > nx || b < a) {
            Py_DECREF(areas);
            PyErr_Format(PyExc_IndexError,
                         "window %zd span [%zd, %zd) out of bounds [0, %zd)",
                         (Py_ssize_t)k, (Py_ssize_t)a, (Py_ssize_t)b,
                         (Py_ssize_t)nx);
            goto done;
        }
    }
    /* Pure-C nested integral loop; drop the GIL (see py_simpson_windows). */
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp r = 0; r < nrows; ++r) {
        const double *row = Yp + r * nx;
        double *orow = ap + r * m;
        for (npy_intp k = 0; k < m; ++k) {
            npy_intp a = sp[k];
            orow[k] = simpson_1d(row + a, xp + a, ep[k] - a);
        }
    }
    Py_END_ALLOW_THREADS
    out = (PyObject *)areas;

done:
    Py_XDECREF(Y);
    Py_XDECREF(x);
    Py_XDECREF(starts);
    Py_XDECREF(stops);
    return out;
}

/* legendre_project(fv, qw, legvander, norm) -> coeffs[k]
 *
 * One fused Gauss-Legendre spectral projection:
 *     coeffs[j] = norm[j] * sum_i  qw[i] * fv[i] * legvander[i, j]
 * that is, norm * ((qw * fv) @ legvander) in a single pass without
 * temporaries. legvander is (nq, k); fv, qw are (nq,); norm is (k,).
 */
static PyObject *py_legendre_project(PyObject *self, PyObject *args)
{
    PyObject *fo, *wo, *vo, *no;
    if (!PyArg_ParseTuple(args, "OOOO", &fo, &wo, &vo, &no)) {
        return NULL;
    }

    PyArrayObject *fv = as_f64(fo, 1);
    PyArrayObject *qw = as_f64(wo, 1);
    PyArrayObject *V = as_f64(vo, 2);
    PyArrayObject *norm = as_f64(no, 1);
    PyObject *out = NULL;
    if (!fv || !qw || !V || !norm) {
        goto done;
    }

    npy_intp nq = PyArray_DIM(V, 0);
    npy_intp k = PyArray_DIM(V, 1);
    if (PyArray_DIM(fv, 0) != nq || PyArray_DIM(qw, 0) != nq) {
        PyErr_SetString(PyExc_ValueError,
                        "fv and qw must have length legvander.shape[0]");
        goto done;
    }
    if (PyArray_DIM(norm, 0) != k) {
        PyErr_SetString(PyExc_ValueError,
                        "norm must have length legvander.shape[1]");
        goto done;
    }

    npy_intp dims[1] = {k};
    PyArrayObject *coeffs = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!coeffs) {
        goto done;
    }

    const double *fp = (const double *)PyArray_DATA(fv);
    const double *wp = (const double *)PyArray_DATA(qw);
    const double *Vp = (const double *)PyArray_DATA(V);
    const double *np_ = (const double *)PyArray_DATA(norm);
    double *cp = (double *)PyArray_DATA(coeffs);

    /* Pure-C projection over borrowed buffers; drop the GIL so a thread pool
     * can run concurrent projections (see py_simpson_windows). */
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp j = 0; j < k; ++j) {
        cp[j] = 0.0;
    }
    /* Row-major accumulation: one streaming pass over the (nq, k) Vandermonde. */
    for (npy_intp i = 0; i < nq; ++i) {
        double wf = wp[i] * fp[i];
        const double *vrow = Vp + i * k;
        for (npy_intp j = 0; j < k; ++j) {
            cp[j] += wf * vrow[j];
        }
    }
    for (npy_intp j = 0; j < k; ++j) {
        cp[j] *= np_[j];
    }
    Py_END_ALLOW_THREADS
    out = (PyObject *)coeffs;

done:
    Py_XDECREF(fv);
    Py_XDECREF(qw);
    Py_XDECREF(V);
    Py_XDECREF(norm);
    return out;
}

static PyMethodDef methods[] = {
    {"simpson_windows", py_simpson_windows, METH_VARARGS,
     "simpson_windows(y, x, starts, stops) -> areas[m]\n"
     "Composite-Simpson integral of y over m index windows [starts[k],stops[k])."},
    {"simpson_windows_rows", py_simpson_windows_rows, METH_VARARGS,
     "simpson_windows_rows(Y, x, starts, stops) -> areas[nrows, m]\n"
     "Per-row composite-Simpson integral of a 2-D Y over the same windows."},
    {"legendre_project", py_legendre_project, METH_VARARGS,
     "legendre_project(fv, qw, legvander, norm) -> coeffs[k]\n"
     "Fused norm * ((qw * fv) @ legvander) Gauss-Legendre spectral projection."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Compiled numeric kernels for dtfit (Simpson windows, Legendre projection).",
    -1,
    methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__native(void)
{
    PyObject *mod = PyModule_Create(&moduledef);
    if (!mod) {
        return NULL;
    }
    import_array();  /* expands to `return NULL;` on failure */
    return mod;
}
