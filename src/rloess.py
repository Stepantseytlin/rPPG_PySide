import warnings

import numpy as np
from skmisc.loess import loess


def rloess(
    y: np.ndarray,
    Smo: int,
    axis: int = -1,
    *,
    degree: int = 2,
    robust: bool = True,
) -> np.ndarray:
    """
    LOESS-сглаживание вдоль указанной оси.

    Parameters
    ----------
    y:
        Конечный массив любой размерности.
    Smo:
        Размер локального окна строго в отсчётах.
    axis:
        Ось времени.
    degree:
        Степень локального полинома.
    robust:
        Использовать robust LOESS family="symmetric".

    Returns
    -------
    np.ndarray
        Массив той же формы, что и y.
    """
    values = np.asarray(y, dtype=np.float64)

    if values.ndim == 0:
        raise ValueError("rloess ожидает массив размерности не меньше 1")

    if not np.all(np.isfinite(values)):
        raise ValueError("rloess получил NaN или inf")

    if isinstance(Smo, bool) or not isinstance(Smo, (int, np.integer)):
        raise TypeError("Smo должен быть целым числом отсчётов")

    axis = int(axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise np.AxisError(axis, values.ndim)

    window = int(Smo)
    length = values.shape[axis]

    minimum_window = degree + 2
    if window < minimum_window:
        raise ValueError(
            f"Smo={window} слишком мал для degree={degree}; "
            f"требуется не меньше {minimum_window}"
        )
    if window > length:
        raise ValueError(
            f"Smo={window} превышает длину сигнала {length}"
        )

    span = window / length
    x = np.arange(length, dtype=np.float64)

    # Переносим временную ось в конец и превращаем остальные
    # измерения в независимые временные ряды.
    moved = np.moveaxis(values, axis, -1)
    rows = moved.reshape(-1, length)
    smoothed = np.empty_like(rows)

    gaussian_fallbacks = 0

    def fit_row(row: np.ndarray, family: str) -> np.ndarray:
        model = loess(
            x,
            row,
            span=span,
            degree=degree,
            surface="direct",
            family=family,
            normalize=False,
        )
        model.fit()

        prediction = np.asarray(
            model.predict(x, stderror=False).values,
            dtype=np.float64,
        )

        if prediction.shape != row.shape:
            raise RuntimeError(
                f"LOESS вернул форму {prediction.shape}, "
                f"ожидалась {row.shape}"
            )
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("LOESS вернул NaN или inf")

        return prediction

    for index, row in enumerate(rows):
        family = "symmetric" if robust else "gaussian"

        try:
            smoothed[index] = fit_row(row, family)

        except Exception as robust_error:
            if not robust:
                raise RuntimeError(
                    f"LOESS не смог обработать ряд {index} "
                    f"с окном {window}"
                ) from robust_error

            # Важно: меняется только robust-режим.
            # Размер окна остаётся тем же.
            try:
                smoothed[index] = fit_row(row, "gaussian")
                gaussian_fallbacks += 1
            except Exception as gaussian_error:
                raise RuntimeError(
                    f"LOESS не смог обработать ряд {index} "
                    f"с фиксированным окном {window}; "
                    f"robust error: {robust_error}"
                ) from gaussian_error

    if gaussian_fallbacks:
        warnings.warn(
            f"Robust LOESS не сработал для {gaussian_fallbacks} рядов; "
            f"использован gaussian LOESS с тем же окном Smo={window}",
            RuntimeWarning,
            stacklevel=2,
        )

    result = smoothed.reshape(moved.shape)
    return result #np.moveaxis(result, -1, axis)
# import numpy as np
# from skmisc.loess import loess


# def rloess(y, Smo, axis=-1):
#     y = np.asarray(y, dtype=float)

#     if y.ndim not in (1, 2):
#         raise ValueError(
#             f"rloess expects 1D or 2D input, got shape {y.shape}"
#         )

#     if Smo <= 0:
#         raise ValueError("Smo должен быть положительным")

#     original_ndim = y.ndim

#     if original_ndim == 1:
#         work = y[None, :]
#     elif axis == 0:
#         work = y.T
#     elif axis == -1:
#         work = y
#     else:
#         raise ValueError("axis должен быть 0 или -1")

#     result = []

#     for y_ in work:
#         if not np.all(np.isfinite(y_)):
#             raise ValueError("rloess получил NaN или inf")

#         x = np.arange(len(y_), dtype=float)

#         # Smo задаётся в отсчётах, span — доля длины сигнала.
#         base_span = Smo / len(y_)
#         base_span = np.clip(
#             base_span,
#             3 / len(y_),   # минимум для degree=2
#             1.0,
#         )

#         # Повторяем исходную идею увеличения окна,
#         # но никогда не выходим за span=1.
#         trial_spans = []

#         for factor in (1, 5, 25, 125):
#             span = min(base_span * factor, 1.0)

#             if not trial_spans or not np.isclose(
#                 span, trial_spans[-1]
#             ):
#                 trial_spans.append(span)

#         prediction = None
#         last_error = None

#         for span in trial_spans:
#             try:
#                 model = loess(
#                     x,
#                     y_,
#                     span=span,
#                     degree=2,
#                     surface="direct",
#                     family="symmetric",
#                     normalize=False,
#                 )
#                 model.fit()

#                 prediction = model.predict(
#                     x,
#                     stderror=False,
#                 ).values
#                 ss = span
#                 break

#             except Exception as exc:
#                 last_error = exc

#         # Последний резервный вариант:
#         # тот же размер окна, но без robust-итераций.
#         if prediction is None:
#             try:
#                 model = loess(
#                     x,
#                     y_,
#                     span=base_span,
#                     degree=2,
#                     surface="direct",
#                     family="gaussian",
#                     normalize=False,
#                 )
#                 model.fit()

#                 prediction = model.predict(
#                     x,
#                     stderror=False,
#                 ).values
#                 ss = base_span
#             except Exception:
#                 # Лучше сохранить исходный сигнал,
#                 # чем неожиданно превращать его в abs(y).
#                 prediction = y_.copy()
#                 ss = 0

#         result.append(prediction)

#     result = np.asarray(result)
#     print(f"final SMO = {ss*len(y_)}, not {Smo}")

#     if original_ndim == 2 and axis == 0:
#         result = result.T
#     return result