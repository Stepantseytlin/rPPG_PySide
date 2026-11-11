import numpy as np
from .segment_timeseries import segment_timeseries
from scipy.linalg import svd as scipy_svd
def stable_svd(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("SVD input contains NaN or infinite values")
    try:
        return scipy_svd(matrix, full_matrices=False, lapack_driver="gesvd")
    except Exception:
        return np.linalg.svd(matrix, full_matrices=False)

def compute_SSA(TS : np.ndarray, depth : int, nET : list[int], overlap : float = 0.98, method : str = 'SSA'):
    """Применение метода спектрального сингулярного анализа "Гусеница"

        Parameters
        ----------
        TS : np.ndarray
            Одномерный массив отсчетов временного ряда.
        depth : int
            Ширина окна (глубина погружения).
        nET : int
            Число эмпирических трендов (ET) eigen triplets для восстановления.
        overlap : float default=0.98
            коэффициент перекрытия векторов траекторной матрицы. Шаг (lagSize) пересчитывается исходя из overlap = (1-overlap)*win
        method : str, default='SSA'
            Метод вычисления АКФ:
            - 'SSA' : стандартный SSA (по умолчанию)
            - 'Med' : медианный метод

        Returns
        -------
        np.ndarray
            Массив восстановленного ряда выбранным числом eigen triplets длины lagS.

    """
    # Сначала сегментируем ряд
    if len(TS.shape)<2:
        TS = TS.reshape(1,TS.shape[0])
    reconstructed_TS = np.zeros((TS.shape))

    for nseg , segment in enumerate(TS):
        TrajectoryMatrix = segment_timeseries(segment, depth, overlap)
        if method == 'SSA':
            # Выполняем SVD на траекторной матрице
            U, S, Vt = stable_svd(TrajectoryMatrix.T)
           # print(U.shape, S.shape, Vt.shape)
            # Выбираем первые nET компонент
            U_n = U[:, nET[0]:nET[1]]
            S_n = np.diag(S[nET[0]:nET[1]])
            Vt_n = Vt[nET[0]:nET[1], :]
            # Восстанавливаем ряд
            Y_recovered = U_n @ S_n @ Vt_n
            # Суммируем антидиагонали
            #reconstructed_TS[nseg] = np.zeros(segment.shape[0])
            #print(Y_recovered.shape)
            # plt.plot(Y_recovered[0,:])
            for i in range(-(Y_recovered.shape[0]-1),Y_recovered.shape[1]-1) :
                reconstructed_TS[nseg,i] = np.mean(np.diag(np.fliplr(Y_recovered), i))
 
            reconstructed_TS[nseg] = np.fliplr(reconstructed_TS[nseg].reshape(1,reconstructed_TS[nseg].shape[-1])).T[:,0]
  

        else:
            raise NotImplementedError("Метод 'Med' пока не реализован")
    return  reconstructed_TS , Y_recovered, S