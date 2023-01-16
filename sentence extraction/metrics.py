"""Thanks to https://github.com/chakki-works/seqeval
Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def f1_score(y_true, y_pred, classi = 4, average='micro', digits=2, suffix=True):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = ['O', 'M']
        >>> y_pred = ['O', 'T']
        >>> f1_score(y_true, y_pred)
        0.50
    """

    ACC = 0.
    ALL = 0.
    T_shots = 0.
    M_shots = 0.
    MT_shots = 0.
    O_shots = 0.
    T_true = 0.
    T_pre = 0.
    M_true = 0.
    M_pre = 0.
    O_true = 0.
    O_pre = 0.
    MT_true = 0.
    MT_pre = 0.

    for label, label_pred in zip(y_true, y_pred):
        str_lab = str(label)
        str_lab_pred = str(label_pred)
        ACC +=  str_lab== str_lab_pred
        ALL = ALL + 1.
        if str_lab == str_lab_pred and str_lab == "T":
            T_shots += 1
        if str_lab == str_lab_pred and str_lab == "M":
            M_shots += 1
        if str_lab == str_lab_pred and str_lab == "MT":
            MT_shots += 1
        if str_lab == str_lab_pred and str_lab == "O":
            O_shots += 1
        if str_lab == "T":
            T_true += 1
        if str_lab_pred == "T":
            T_pre += 1
        if str_lab == "O":
            O_true += 1
        if str_lab_pred == "O":
            O_pre += 1
        if str_lab == "M":
            M_true += 1
        if str_lab_pred == "M":
            M_pre += 1
        if str_lab == "MT":
            MT_true += 1
        if str_lab_pred == "MT":
            MT_pre += 1
    PREC_T = 0
    REC_T = 0
    F1_T = 0
    PREC_M = 0
    REC_M = 0
    F1_M = 0
    PREC_MT = 0
    REC_MT = 0
    F1_MT = 0
    PREC_O = 0
    REC_O = 0
    F1_O = 0
    Macro_P = 0
    Micro_P = 0
    Macro_R = 0
    Micro_R = 0
    Macro_F1 = 0
    Micro_F1 = 0
    if T_true > 0 and T_pre > 0 and T_shots > 0:
        PREC_T = float(T_shots) / float(T_pre)
        REC_T = float(T_shots) / float(T_true)
        F1_T = 2 * PREC_T * REC_T / (PREC_T + REC_T)
    if M_true > 0 and M_pre > 0 and M_shots > 0:
        PREC_M = float(M_shots) / float(M_pre)
        REC_M = float(M_shots) / float(M_true)
        F1_M = 2 * PREC_M * REC_M / (PREC_M + REC_M)
    if MT_true > 0 and MT_pre > 0 and MT_shots > 0:
        PREC_MT = float(MT_shots) / float(MT_pre)
        REC_MT = float(MT_shots) / float(MT_true)
        F1_MT = 2 * PREC_MT * REC_MT / (PREC_MT + REC_MT)
    if MT_true > 0 and MT_pre > 0 and MT_shots > 0:
        PREC_MT = float(MT_shots) / float(MT_pre)
        REC_MT = float(MT_shots) / float(MT_true)
        F1_MT = 2 * PREC_MT * REC_MT / (PREC_MT + REC_MT)
    if O_true > 0 and O_pre > 0 and O_shots > 0:
        PREC_O = float(O_shots) / float(O_pre)
        REC_O = float(O_shots) / float(O_true)
        F1_O = 2 * PREC_O * REC_O / (PREC_O + REC_O)
    acc = float(ACC) / float(ALL)
    if classi == 2:
        Macro_P = (PREC_MT + PREC_O) / 2.0
        Macro_R = (REC_MT + REC_O) / 2.0
        if Macro_P>0 and Macro_R>0:
            Macro_F1 = 2 * Macro_P * Macro_R / (Macro_P + Macro_R)
        else:
            Macro_F1 = 0
        Micro_P = (MT_shots + O_shots) / (MT_pre + O_pre)
        Micro_R = (MT_shots + O_shots) / (MT_true + O_true)
        if Micro_P>0 and Micro_R>0:
            Micro_F1 = 2 * Micro_P * Micro_R / (Micro_P + Micro_R)
        else:
            Micro_F1 = 0
    if classi == 3:
        Macro_P = (PREC_MT + PREC_M + PREC_T) / 3.0
        Macro_R = (REC_MT + REC_M + REC_T) / 3.0
        Macro_F1 = 2 * Macro_P * Macro_R / (Macro_P + Macro_R)
        Micro_P = (MT_shots + M_shots + T_shots) / (MT_pre + M_pre + T_pre)
        Micro_R = (MT_shots + M_shots + T_shots) / (MT_true + M_true + T_true)
        Micro_F1 = 2 * Micro_P * Micro_R / (Micro_P + Micro_R)
    if classi == 4:
        Macro_P = (PREC_MT + PREC_M + PREC_T + PREC_O) / 4.0
        Macro_R = (REC_MT + REC_M + REC_T + REC_O) / 4.0

        Macro_F1 = 2 * Macro_P * Macro_R / (Macro_P + Macro_R)
        Micro_P = (MT_shots + M_shots + T_shots + O_shots) / (MT_pre + M_pre + T_pre + O_pre)
        Micro_R = (MT_shots + M_shots + T_shots + O_shots) / (MT_true + M_true + T_true + O_true)
        Micro_F1 = 2 * Micro_P * Micro_R / (Micro_P + Micro_R)


    print(100 * acc)
    print(100 * Macro_P)
    print(100 * Macro_R)
    print(100 * Macro_F1)
    print(100 * Micro_P)
    print(100 * Micro_R)
    print(100 * Micro_F1)
    print(100 * PREC_T)
    print(100 * REC_T)
    print(100 * F1_T)
    print(100 * PREC_M)
    print(100 * REC_M)
    print(100 * F1_M)
    print(100 * PREC_MT)
    print(100 * REC_MT)
    print(100 * F1_MT)
    print(100 * PREC_O)
    print(100 * REC_O)
    print(100 * F1_O)

    return {"acc": 100 * acc, 'Macro_P': 100 * Macro_P, 'Macro_R': 100 * Macro_R, "Macro_F1": 100 * Macro_F1,
            "Micro_P": 100 * Micro_P, "Micro_R": 100 * Micro_R, "Micro_F1": 100 * Micro_F1, "PREC_T": 100 * PREC_T,
            "REC_T": 100 * REC_T, "F1_T": 100 * F1_T, "PREC_T": 100 * PREC_T, "REC_T": 100 * REC_T, "F1_T": 100 * F1_T,
            "PREC_M": 100 * PREC_M, "REC_M": 100 * REC_M, "F1_M": 100 * F1_M, "PREC_MT": 100 * PREC_MT,
            "REC_MT": 100 * REC_MT, "F1_MT": 100 * F1_MT, "PREC_O": 100 * PREC_O, "REC_O": 100 * REC_O, "F1_O": 100 * F1_O}

