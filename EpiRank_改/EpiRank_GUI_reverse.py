# coding=utf-8
"""
EpiRank GUI — Epidemic Risk Analysis System (PySide6)

Reverse Sequential-day 版本：一天一次迭代，日内先晚后早传播，日末只加一次外部因子。

核心公式：
    ER_{t+1} = (1-d)·e + d · W^T · W · ER_t

其中：
    W   = 回程（evening）方向列归一化 OD 矩阵
    W^T = 去程（morning）方向列归一化转置 OD 矩阵
    e   = 均匀外部因子 (1/N)
    d   = 阻尼系数

收敛性由 Perron-Frobenius 定理保证（M_day = (1-d)E + d·W^T·W 严格正、列随机）。
分类使用 head/tail breaks (Jiang 2013)。

Note: 无 daytime 参数。一天只有一个 EpiRank 结果。

GUI Tab 布局：
    0  Results Table
    1  Network Map
    2  Core Classification
    3  Evaluation Summary
    4  Commuter Flow
    5  Frequency Distributions
    6  EpiRank Distribution
    7  EpiRank vs Disease
    8  Index Comparison
    9  Disease Map
   10  EpiRank Map
   11  EpiRank vs Disease Map
   12  Log
   13  Sensitivity Analysis
   14  Transient Dynamics (M1)
   15  Attribution (M2)
"""

import sys
import os
import json
import numpy as np
import networkx as nx
from scipy import stats as st
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QFileDialog, QTextEdit, QTabWidget, QProgressBar,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QFormLayout, QComboBox, QStatusBar, QScrollArea, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QAction

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import font_manager

# ---- CJK 字体配置 ----
_CJK_FONT_CANDIDATES = [
    # ---- Windows ----
    'Microsoft YaHei', 'Microsoft JhengHei',
    'SimHei', 'SimSun', 'MingLiU', 'PMingLiU',
]
_cjk_font_found = None
_available_font_names = {f.name for f in font_manager.fontManager.ttflist}
for _candidate in _CJK_FONT_CANDIDATES:
    if _candidate in _available_font_names:
        _cjk_font_found = _candidate
        break

if _cjk_font_found:
    matplotlib.rcParams['font.sans-serif'] = [_cjk_font_found] + matplotlib.rcParams.get('font.sans-serif', [])
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f"[Matplotlib] Using CJK font: {_cjk_font_found}")
else:
    print("[Matplotlib] WARNING: No CJK font found.")

# ---- 数据键名常量 ----
KEY_POST_CODE            = 'post_code'
KEY_DB_ID                = 'db_ID'
KEY_COUNTY               = 'county'
KEY_TOWN                 = 'town'
KEY_POS_XY               = 'pos_xy'
KEY_POPULATION           = 'population'
KEY_AREA                 = 'area'
KEY_DENSITY              = 'density'
KEY_NORMALIZED_DENSITY   = 'normalized_density'
KEY_AGE_0_14             = 'age_0_14'
KEY_AGE_15_64            = 'age_15_64'
KEY_AGE_65               = 'age_65'
KEY_LOCAL_COMMUTER_TYPE1 = 'local_commuter_type1'
KEY_OUT_COMMUTER_TYPE1   = 'out_commuter_type1'
KEY_IN_COMMUTER_TYPE1    = 'in_commuter_type1'
KEY_COMMUTER_TYPE1       = 'commuter_type1'
KEY_RAILROAD_ZONE        = 'railroad_zone'
KEY_FLU_TOTAL_CASES      = 'flu_total_cases'
KEY_EV_AVERAGE_CASES     = 'EV_average_cases'
KEY_SARS_TOTAL_CASES     = 'sars_total_cases'

# 大台北都会区 48 乡镇市
GTAIPEI_DB_IDS = set(range(0, 29)) | set(range(303, 310)) | set(range(330, 397))


# ============================================================
# 通用工具
# ============================================================

def sorted_map(mapping):
    """依值降序、键升序排序字典。"""
    return sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))


def build_basic_table_of_towns(town_data, path='bs.xlsx', sheet='town_data',
                                number_of_sub_towns=409, row_base=2):
    """从 bs.xlsx 载入乡镇基本资料。"""
    wb = load_workbook(path, data_only=True)
    s = wb[sheet]
    old_town = None
    for row_idx in range(number_of_sub_towns):
        r = row_idx + row_base
        cell_val = s.cell(row=r, column=1).value
        if cell_val is None:
            continue
        db_ID              = int(cell_val)
        county             = s.cell(row=r, column=2).value
        town               = s.cell(row=r, column=3).value
        pos_xy             = (round(float(s.cell(row=r, column=7).value or 0), 2),
                              round(float(s.cell(row=r, column=8).value or 0), 2))
        raw_population     = s.cell(row=r, column=9).value or 0
        sub_percentage     = float(s.cell(row=r, column=10).value or 0)
        area               = float(s.cell(row=r, column=12).value or 0)
        density            = float(s.cell(row=r, column=13).value or 0)
        normalized_density = float(s.cell(row=r, column=14).value or 0)
        age_0_14           = float(s.cell(row=r, column=15).value or 0)
        age_15_64          = float(s.cell(row=r, column=16).value or 0)
        age_65             = float(s.cell(row=r, column=17).value or 0)
        population         = (float(raw_population / sub_percentage)
                              if sub_percentage > 0 else float(raw_population))
        if town != old_town:
            town_data[db_ID] = {
                KEY_COUNTY: county, KEY_TOWN: town, KEY_POS_XY: pos_xy,
                KEY_POPULATION: population, KEY_AREA: area,
                KEY_DENSITY: density, KEY_NORMALIZED_DENSITY: normalized_density,
                KEY_AGE_0_14: age_0_14, KEY_AGE_15_64: age_15_64, KEY_AGE_65: age_65,
                KEY_LOCAL_COMMUTER_TYPE1: 0, KEY_OUT_COMMUTER_TYPE1: 0,
                KEY_IN_COMMUTER_TYPE1: 0, KEY_RAILROAD_ZONE: 0
            }
        old_town = town
    wb.close()


def build_flu_reported_cases(town_data, path='Flu.xlsx', sheet='2009',
                              number_of_towns=353, row_base=2):
    """载入 2009 流感病例数。"""
    wb = load_workbook(path, data_only=True)
    s = wb[sheet]
    check_list = {}
    for row_idx in range(number_of_towns):
        r = row_idx + row_base
        county = s.cell(row=r, column=1).value
        town_name = s.cell(row=r, column=2).value
        if not check_list:
            for db_ID in town_data.keys():
                check_list[(town_data[db_ID][KEY_COUNTY], town_data[db_ID][KEY_TOWN])] = db_ID
        db_ID = check_list.get((county, town_name), None)
        if db_ID is not None:
            raw = s.cell(row=r, column=3).value
            town_data[db_ID][KEY_FLU_TOTAL_CASES] = int(raw) if raw is not None else 0
    wb.close()


def build_ev_reported_cases(town_data, path='ev.xlsx', sheet='2000_2008',
                             number_of_towns=353, row_base=2):
    """载入肠病毒平均病例数。"""
    wb = load_workbook(path, data_only=True)
    s = wb[sheet]
    check_list = {}
    for row_idx in range(number_of_towns):
        r = row_idx + row_base
        county = s.cell(row=r, column=1).value
        town_name = s.cell(row=r, column=2).value
        if not check_list:
            for db_ID in town_data.keys():
                check_list[(town_data[db_ID][KEY_COUNTY], town_data[db_ID][KEY_TOWN])] = db_ID
        db_ID = check_list.get((county, town_name), None)
        if db_ID is not None:
            raw = s.cell(row=r, column=3).value
            town_data[db_ID][KEY_EV_AVERAGE_CASES] = float(raw) if raw is not None else 0.0
    wb.close()


def build_sars_reported_cases(town_data, path='SARS.xlsx', sheet='2003',
                               number_of_towns=353, row_base=2):
    """载入 2003 SARS 病例数（仅用于大台北相关性分析）。"""
    wb = load_workbook(path, data_only=True)
    s = wb[sheet]
    check_list = {}
    for row_idx in range(number_of_towns):
        r = row_idx + row_base
        county = s.cell(row=r, column=1).value
        town_name = s.cell(row=r, column=2).value
        if not check_list:
            for db_ID in town_data.keys():
                check_list[(town_data[db_ID][KEY_COUNTY], town_data[db_ID][KEY_TOWN])] = db_ID
        db_ID = check_list.get((county, town_name), None)
        if db_ID is not None:
            raw = s.cell(row=r, column=3).value
            town_data[db_ID][KEY_SARS_TOTAL_CASES] = int(raw) if raw is not None else 0
    wb.close()


def build_commuting_network(g, town_data, path='cn.xlsx', sheet='353C',
                             number_of_towns=353, row_base=6, col_base=6):
    """从 cn.xlsx 建立 353×353 通勤网络。"""
    wb = load_workbook(path, read_only=False, data_only=True)
    s = wb[sheet]
    cache = {}
    for row_idx in range(number_of_towns):
        r = row_idx + row_base
        raw_seq = s.cell(row=r, column=1).value
        if raw_seq is None:
            continue
        row_seq_no = int(raw_seq)
        if row_seq_no in cache:
            row_code, row_db_ID = cache[row_seq_no]
        else:
            row_code = str(s.cell(row=r, column=2).value or '')
            raw_db = s.cell(row=r, column=3).value
            if raw_db is None:
                continue
            row_db_ID = int(raw_db)
            cache[row_seq_no] = (row_code, row_db_ID)

        for col_idx in range(number_of_towns):
            c = col_idx + col_base
            raw_col_seq = s.cell(row=1, column=c).value
            if raw_col_seq is None:
                continue
            col_seq_no = int(raw_col_seq)
            if col_seq_no in cache:
                col_code, col_db_ID = cache[col_seq_no]
            else:
                col_code = str(s.cell(row=2, column=c).value or '')
                raw_col_db = s.cell(row=3, column=c).value
                if raw_col_db is None:
                    continue
                col_db_ID = int(raw_col_db)
                cache[col_seq_no] = (col_code, col_db_ID)

            raw_commuters = s.cell(row=r, column=c).value
            commuters = int(raw_commuters) if raw_commuters is not None else 0

            if commuters > 0:
                if row_db_ID not in town_data or col_db_ID not in town_data:
                    continue
                if not g.has_node(row_seq_no):
                    g.add_node(row_seq_no, post_code=row_code, db_ID=row_db_ID,
                               posx=town_data[row_db_ID][KEY_POS_XY][0],
                               posy=town_data[row_db_ID][KEY_POS_XY][1])
                if not g.has_node(col_seq_no):
                    g.add_node(col_seq_no, post_code=col_code, db_ID=col_db_ID,
                               posx=town_data[col_db_ID][KEY_POS_XY][0],
                               posy=town_data[col_db_ID][KEY_POS_XY][1])
                g.add_edge(row_seq_no, col_seq_no, weight=float(commuters),
                           commuter_type1=float(commuters))
                town_data[row_db_ID][KEY_OUT_COMMUTER_TYPE1] += commuters
                town_data[col_db_ID][KEY_IN_COMMUTER_TYPE1] += commuters
    wb.close()


def get_pearson_cor(dic1, dic2):
    """Pearson 相关系数。"""
    assert set(dic1.keys()) == set(dic2.keys()), \
        f"Key mismatch: {len(dic1)} vs {len(dic2)} keys"
    keys = list(dic1.keys())
    if len(keys) < 3:
        return (float('nan'), float('nan'))
    n1 = [dic1[k] for k in keys]
    n2 = [dic2[k] for k in keys]
    r, p = st.pearsonr(n1, n2)
    return round(r, 6), round(p, 6)


def get_spearman_cor(dic1, dic2):
    """Spearman 等级相关系数。"""
    assert set(dic1.keys()) == set(dic2.keys()), \
        f"Key mismatch: {len(dic1)} vs {len(dic2)} keys"
    keys = list(dic1.keys())
    if len(keys) < 3:
        return (float('nan'), float('nan'))
    n1 = [dic1[k] for k in keys]
    n2 = [dic2[k] for k in keys]
    r, p = st.spearmanr(n1, n2)
    return round(r, 6), round(p, 6)


def get_kendalltau_cor(dic1, dic2):
    """Kendall's tau 相关系数。"""
    assert set(dic1.keys()) == set(dic2.keys()), \
        f"Key mismatch: {len(dic1)} vs {len(dic2)} keys"
    keys = list(dic1.keys())
    if len(keys) < 3:
        return (float('nan'), float('nan'))
    n1 = [dic1[k] for k in keys]
    n2 = [dic2[k] for k in keys]
    r, p = st.kendalltau(n1, n2)
    return round(r, 6), round(p, 6)


# ============================================================
# Head/Tail Breaks 分类
# ============================================================

LEVEL_COLORS = {
    'NC':    '#3a8f3e',
    'C-III': '#c8b840',
    'C-II':  '#e07830',
    'C-I':   '#cc2020',
}
LEVEL_ORDER = ['NC', 'C-III', 'C-II', 'C-I']

OVERLAP_LEVEL_SETS = [
    ('C-I 核心',           {'C-I'}),
    ('C-II 核心',          {'C-II'}),
    ('C-III 核心',         {'C-III'}),
    ('全部核心 (I+II+III)', {'C-I', 'C-II', 'C-III'}),
    ('NC 非核心',          {'NC'}),
]


def compute_overlap_metrics(labels_a, labels_b, positive_levels):
    """Compute spatial overlap between two sets of core labels.

    labels_a / labels_b : { db_id_str: {'level': 'C-I'|'C-II'|'C-III'|'NC', ...} }
    positive_levels     : set, e.g. {'C-I'} or {'C-I','C-II','C-III'}
    """
    common = set(labels_a.keys()) & set(labels_b.keys())
    set_a = {k for k in common if labels_a[k]['level'] in positive_levels}
    set_b = {k for k in common if labels_b[k]['level'] in positive_levels}
    inter = set_a & set_b
    union = set_a | set_b
    n_a, n_b = len(set_a), len(set_b)
    n_i, n_u = len(inter), len(union)
    iou   = n_i / n_u if n_u > 0 else 0.0
    dice  = 2 * n_i / (n_a + n_b) if (n_a + n_b) > 0 else 0.0
    cov_a = n_i / n_a if n_a > 0 else 0.0
    cov_b = n_i / n_b if n_b > 0 else 0.0
    return {
        'n_a': n_a, 'n_b': n_b, 'n_i': n_i, 'n_u': n_u,
        'iou': iou, 'dice': dice, 'cov_a': cov_a, 'cov_b': cov_b,
    }


def head_tail_breaks(values, n_breaks=3):
    """递归以平均值分裂（Jiang 2013）。"""
    breaks = []
    head = values.copy()
    for _ in range(n_breaks):
        if len(head) < 2:
            break
        m = head.mean()
        breaks.append(m)
        head = head[head > m]
    return sorted(breaks)


def classify_by_breaks(values, breaks):
    """依 head/tail breaks 分类為 NC / C-III / C-II / C-I。"""
    if len(breaks) == 0:
        return ['NC'] * len(values)
    while len(breaks) < 3:
        breaks = list(breaks) + [breaks[-1]]

    labels = []
    for v in values:
        if v <= breaks[0]:
            labels.append('NC')
        elif v <= breaks[1]:
            labels.append('C-III')
        elif v <= breaks[2]:
            labels.append('C-II')
        else:
            labels.append('C-I')
    return labels


# ============================================================
# EpiRank 核心：Reverse Sequential-day 模式
# ============================================================

def compute_epidemic_risk(g, town_data, d, number_of_loops=5000,
                           progress_callback=None,
                           record_trajectory=False,
                           record_attribution=False):
    """Reverse Sequential-day EpiRank 核心（M1 瞬态 + M2 归因）。

    日内先晚后早，日末只加一次外部因子：
        ER_{t+1} = (1-d)·e + d · W^T · (W · ER_t)

    收敛性：M_day = (1-d)E + d·W^T·W 严格正、列随机。

    Parameters
    ----------
    record_trajectory : bool
        M1：记录迭代轨迹并计算瞬态指标。
    record_attribution : bool
        M2：累积 morning/evening 风险流矩阵，得到方向性归因。
    """
    Ncount = g.order()

    # ---- Phase A: 建立并列归一化 OD 矩阵 ----
    CN_C = nx.to_numpy_array(g, weight=KEY_COMMUTER_TYPE1)
    CN_T = CN_C.copy()

    csum = CN_T.sum(axis=0)
    CN = np.zeros((Ncount, Ncount))
    for i in range(CN_T.shape[0]):
        for j in range(CN_T.shape[1]):
            s = float(csum[j])
            if s > 0:
                CN[i, j] = float(CN_T[i, j]) / s

    ODt = CN_T.T
    osum = ODt.sum(axis=0)
    CNt = np.zeros((Ncount, Ncount))
    for i in range(ODt.shape[0]):
        for j in range(ODt.shape[1]):
            s = float(osum[j])
            if s > 0:
                CNt[i, j] = float(ODt[i, j]) / s

    # ---- Phase B: Reverse Sequential-day 迭代 ----
    other_factors = np.ones((Ncount, 1)) / float(Ncount)
    epidemic_risk = np.ones((Ncount, 1)) / float(Ncount)

    trajectory = [] if record_trajectory else None
    F_morning = np.zeros((Ncount, Ncount)) if record_attribution else None
    F_evening = np.zeros((Ncount, Ncount)) if record_attribution else None

    iterations = 0
    for i in range(number_of_loops):
        old_er = epidemic_risk.copy()

        # 第一阶段：Evening
        er_evening = CN @ epidemic_risk

        if record_attribution:
            F_evening += CN * epidemic_risk.T

        # 第二阶段：Morning
        morning_contrib = CNt @ er_evening

        if record_attribution:
            F_morning += CNt * er_evening.T

        # 日末加入外部因子
        epidemic_risk = (
            (1.0 - d) * other_factors
            + d * morning_contrib
        )

        if record_trajectory:
            trajectory.append(epidemic_risk.flatten().copy())

        iterations = i + 1
        if np.allclose(epidemic_risk, old_er, atol=1e-12):
            break

        if progress_callback and (i % 50 == 0 or i == number_of_loops - 1):
            progress_callback(i + 1, number_of_loops)

    # ══════════════════════════════════════════════════════════
    # M1: 从轨迹派生瞬态指标
    # ══════════════════════════════════════════════════════════
    transient_metrics = None
    if record_trajectory and len(trajectory) > 2:
        traj = np.array(trajectory)
        T, N = traj.shape
        er_star = traj[-1]

        theta = 0.5
        arrival_time = np.full(N, np.inf)
        for i in range(N):
            if er_star[i] > 1e-15:
                hits = np.where(traj[:, i] >= theta * er_star[i])[0]
                if len(hits) > 0:
                    arrival_time[i] = float(hits[0])

        K = min(30, T)
        t_axis = np.arange(K, dtype=float)
        velocity = np.zeros(N)
        for i in range(N):
            y = traj[:K, i]
            if y.std() > 1e-12:
                velocity[i] = np.polyfit(t_axis, y, 1)[0]

        E_uniform = np.ones((N, N)) / float(N)
        M_day = (1.0 - d) * E_uniform + d * (CNt @ CN)
        eigvals = np.linalg.eigvals(M_day)

        # ---- 绝对谱隙（保留，用于混合速度报告）----
        abs_eig = np.sort(np.abs(eigvals))[::-1]
        abs_lambda1 = float(abs_eig[0])
        abs_lambda2 = float(abs_eig[1]) if len(abs_eig) > 1 else 0.0
        spectral_gap = 1.0 - abs_lambda2 / abs_lambda1 if abs_lambda1 > 1e-12 else 0.0

        # ---- 经典 Kemeny 常数 ----
        # K = Σ_{r≥2} 1 / (1 - λ_r)，约定 m_ii = 0
        # 1. 找出 Perron 根（最接近 1 的那个特征值）并从谱中剔除
        idx_perron = int(np.argmin(np.abs(eigvals - 1.0)))
        non_perron = np.delete(eigvals, idx_perron)

        # 2. 剔除后可能仍有接近 1 的数值噪声，做一次数值清理
        non_perron = non_perron[np.abs(1.0 - non_perron) > 1e-12]

        # 3. 求和；复数共轭对会自动抵消虚部，取实部即可
        if non_perron.size == 0:
            kemeny_constant = float('inf')
        else:
            kemeny_constant = float(np.real(np.sum(1.0 / (1.0 - non_perron))))

        transient_metrics = {
            'trajectory': traj,
            'arrival_time': arrival_time,
            'velocity': velocity,
            'spectral_gap': spectral_gap,
            'kemeny_constant': kemeny_constant,
            'abs_lambda1': abs_lambda1,
            'abs_lambda2': abs_lambda2,
            'er_star': er_star,
        }

    # ══════════════════════════════════════════════════════════
    # M2: 方向性风险归因
    # ══════════════════════════════════════════════════════════
    attribution_data = None
    if record_attribution:
        F_net = F_morning + F_evening
        net_inflow = F_net.sum(axis=1)
        net_outflow = F_net.sum(axis=0)

        total_flow = F_net.sum()
        if total_flow > 0:
            F_net_norm = F_net / total_flow
        else:
            F_net_norm = F_net

        # ── 闭式解校验 ──
        I = np.eye(Ncount)
        A = I - d * (CNt @ CN)
        try:
            er_star_closed = np.linalg.solve(A, (1.0 - d) * other_factors)
            er_star_closed = er_star_closed.flatten()
            s = er_star_closed.sum()
            if s > 0:
                er_star_closed = er_star_closed / s
            er_iter = epidemic_risk.flatten()
            closed_form_residual = float(np.max(np.abs(er_star_closed - er_iter)))
        except np.linalg.LinAlgError:
            er_star_closed = None
            closed_form_residual = float('nan')

        # ── 每县 top-k 风险源 ──
        K_TOP = 5
        top_sources = {}
        for i in range(Ncount):
            row = F_net[i, :]
            row_no_self = row.copy()
            row_no_self[i] = 0.0
            top_idx = np.argsort(-row_no_self)[:K_TOP]
            top_sources[i] = [(int(j), float(row_no_self[j]))
                              for j in top_idx if row_no_self[j] > 0]

        attribution_data = {
            'F_morning': F_morning,
            'F_evening': F_evening,
            'F_net': F_net,
            'F_net_norm': F_net_norm,
            'net_inflow': net_inflow,
            'net_outflow': net_outflow,
            'top_sources': top_sources,
            'closed_form_residual': closed_form_residual,
            'er_star_closed': er_star_closed,
        }

    return epidemic_risk, iterations, CN_C, transient_metrics, attribution_data


# ============================================================
# M1 / M2 指标日志格式化
# ============================================================
#
# transient_metrics / attribution_data 内含 400 维向量与 400×400 矩阵，
# 直接整份打印会淹没 Log 页。此处对每项输出：
#   (1) 统计摘要（n / finite / min / p50 / mean / max）
#   (2) Top-N 排名表（自动跳过 inf/nan）
#   (3) 矩阵则额外列出权重最大的 N 条边
# N 由 LOG_TOP_N 控制。
# ============================================================

LOG_TOP_N = 10        # 各类排名表输出条数
LOG_TRAJ_FULL = 20    # 轨迹逐轮表：轮数 <= 此值时全部输出，否则只给首尾各 5 轮


def _node_name(g, town_data, node):
    """节点标签：'town [db_ID]'。"""
    db_ID = g.nodes[node][KEY_DB_ID]
    td = town_data[db_ID]
    return f"{td[KEY_TOWN] or td[KEY_COUNTY]} [db {db_ID}]"


def _stat_line(name, values):
    """一行统计摘要；inf/nan 不参与统计但会计数。"""
    a = np.asarray(values, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return f"    {name:<26} n={a.size:<5} (no finite value)"
    return (f"    {name:<26} n={a.size:<5} finite={finite.size:<5} "
            f"min={finite.min():.6g}  p50={np.median(finite):.6g}  "
            f"mean={finite.mean():.6g}  max={finite.max():.6g}")


def _append_rank_table(lines, title, nodes, g, town_data, values,
                       largest=True, top_n=LOG_TOP_N, fmt='{:+.6g}'):
    """输出一个 Top-N 排名表（自动跳过 inf/nan）。"""
    a = np.asarray(values, dtype=float).ravel()
    lines.append(title)
    order = np.argsort(-a if largest else a, kind='stable')
    shown = 0
    for idx in order:
        if not np.isfinite(a[idx]):
            continue
        shown += 1
        lines.append(f"      #{shown:<3d} node={str(nodes[idx]):<6s} "
                     f"{fmt.format(a[idx]):>15s}  "
                     f"{_node_name(g, town_data, nodes[idx])}")
        if shown >= top_n:
            break
    if shown == 0:
        lines.append('      (no finite value)')
    lines.append('')


def _append_matrix_edges(lines, name, M, nodes, g, town_data, top_n=LOG_TOP_N):
    """输出一个风险流矩阵的摘要与权重最大的 top_n 条边。

    方向约定：矩阵以「列 = 风险流出方（来源）」存储，即 M[i, j] 表示风险
    由 j 流向 i，故边写作 nodes[j] -> nodes[i]。

    对角线 F[i, i] 是本地留存的风险，会随迭代单调累积而压过所有跨乡镇边，
    故摘要与排名皆在置零对角线后的矩阵上进行。
    """
    a = np.asarray(M, dtype=float)
    off = a.copy()
    np.fill_diagonal(off, 0.0)
    lines.append(f'    {name}: shape={off.shape}  nnz={int((off > 0).sum())}  '
                 f'sum={off.sum():.6g}  min={off.min():.6g}  '
                 f'max={off.max():.6g}  (diagonal excluded)')
    flat = off.ravel()
    if flat.size == 0 or flat.max() <= 0:
        lines.append('      (no positive edge)')
        lines.append('')
        return
    k = min(top_n, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    n_rows = a.shape[0]
    lines.append(f'      top {k} edges (src -> dst):')
    for rank, fi in enumerate(idx, 1):
        i, j = divmod(int(fi), n_rows)
        lines.append(f"        #{rank:<3d} {str(nodes[j]):>5s} -> {str(nodes[i]):<5s} "
                     f"{flat[fi]:>15.6g}   "
                     f"{_node_name(g, town_data, nodes[j])} -> "
                     f"{_node_name(g, town_data, nodes[i])}")
    lines.append('')


def format_m1_m2_log(nodes, g, town_data, transient_metrics, attribution_data,
                     top_n=LOG_TOP_N):
    """把 M1（瞬态动力学）与 M2（方向性归因）指标编排为可读文本。

    nodes 必须与 compute_epidemic_risk() 中矩阵的行/列顺序一致
    （即 list(g.nodes())），否则节点名与数值会错位。
    """
    lines = ['=' * 78,
             'M1 / M2 指标明细',
             '=' * 78, '']

    # ── M1: 瞬态动力学 ──
    if transient_metrics is None:
        lines.append('[M1] 未记录瞬态轨迹 (record_trajectory=False)')
    else:
        tm = transient_metrics
        traj = tm['trajectory']
        T, N = traj.shape
        lines.append(f'[M1] 瞬态动力学  T={T} iterations, N={N} nodes')
        lines.append('')

        lines.append('  谱性质'
                     '（abs gap = 1-|λ₂|/|λ₁| 只看 λ₂；'
                     'Kemeny = Σ_{r≥2} 1/(1-λ_r) 用到完整谱）')
        lines.append(_stat_line('abs(lambda1)', [tm['abs_lambda1']]))
        lines.append(_stat_line('abs(lambda2)', [tm['abs_lambda2']]))
        lines.append(f"    {'abs_spectral_gap 1-|l2|/|l1|':<26} "
                     f"{tm['spectral_gap']:.6g}")
        kem = tm['kemeny_constant']
        lines.append(f"    {'kemeny_constant sum 1/(1-lr)':<26} "
                     f"{'inf' if not np.isfinite(kem) else format(kem, '.6g')}")
        lines.append('')

        # 轨迹
        lines.append(f'  轨迹 trajectory  shape={traj.shape}')
        lines.append(f"    {'t=0 sum':<26} {traj[0].sum():.10f}")
        lines.append(f"    {'t=T-1 sum':<26} {traj[-1].sum():.10f}")
        er_star = np.asarray(tm['er_star'], dtype=float)
        dev = np.abs(traj[-1] - er_star)
        lines.append(f"    {'max|traj[-1]-ER*|':<26} {dev.max():.6g}")
        if T <= LOG_TRAJ_FULL:
            rows = list(range(T))
        else:
            rows = list(range(5)) + [None] + list(range(T - 5, T))
        lines.append('    per-iteration  mean / min / max:')
        for t in rows:
            if t is None:
                lines.append('      ...')
                continue
            col = traj[t]
            lines.append(f"      t={t:<5d} mean={col.mean():.6e}  "
                         f"min={col.min():.6e}  max={col.max():.6e}")
        if T > LOG_TRAJ_FULL:
            lines.append(f'      (only first/last 5 of {T} iterations shown)')
        lines.append('')

        # 到达时间
        at = np.asarray(tm['arrival_time'], dtype=float)
        lines.append('  到达时间 arrival_time  (首次满足 ER_t >= 0.5 * ER* 的迭代号)')
        lines.append(_stat_line('arrival_time', at))
        lines.append(f"    {'never reached (inf)':<26} {int(np.isinf(at).sum())}")
        _append_rank_table(lines, f'  最快到达 top {top_n} (arrival_time 最小):',
                           nodes, g, town_data, at, largest=False,
                           top_n=top_n, fmt='{:.1f}')

        # 风险速度
        vel = np.asarray(tm['velocity'], dtype=float)
        lines.append('  风险速度 velocity  (前 30 步最小二乘斜率)')
        lines.append(_stat_line('velocity', vel))
        _append_rank_table(lines, f'  速度最高 top {top_n} (velocity 最大):',
                           nodes, g, town_data, vel, largest=True, top_n=top_n)
        _append_rank_table(lines, f'  速度最低 top {top_n} (velocity 最小):',
                           nodes, g, town_data, vel, largest=False, top_n=top_n)

    lines.append('')

    # ── M2: 方向性风险归因 ──
    if attribution_data is None:
        lines.append('[M2] 未记录归因数据 (record_attribution=False)')
    else:
        ad = attribution_data
        lines.append('[M2] 方向性风险归因')
        lines.append('')
        lines.append(f"    {'closed_form_residual':<26} "
                     f"{ad['closed_form_residual']:.3e}")
        lines.append('')

        # 净流入 / 净流出
        n_in = np.asarray(ad['net_inflow'], dtype=float)
        n_out = np.asarray(ad['net_outflow'], dtype=float)
        lines.append('  净流入 net_inflow  (F_net 按行汇总 = 流入本县的风险)')
        lines.append(_stat_line('net_inflow', n_in))
        lines.append('  净流出 net_outflow (F_net 按列汇总 = 本县流出的风险)')
        lines.append(_stat_line('net_outflow', n_out))
        bal = n_in - n_out
        lines.append(f"    {'净流入-净流出':<26} "
                     f"sum={bal.sum():.6g}  mean={bal.mean():.6g}  "
                     f"max={bal.max():.6g}  min={bal.min():.6g}")
        lines.append('')
        _append_rank_table(lines, f'  净流入最高 top {top_n}:',
                           nodes, g, town_data, n_in, largest=True, top_n=top_n)
        _append_rank_table(lines, f'  净流出最高 top {top_n}:',
                           nodes, g, town_data, n_out, largest=True, top_n=top_n)
        _append_rank_table(lines, f'  净流入-净流出最高 top {top_n} (净接收者):',
                           nodes, g, town_data, bal, largest=True, top_n=top_n)
        _append_rank_table(lines, f'  净流入-净流出最低 top {top_n} (净输出者):',
                           nodes, g, town_data, bal, largest=False, top_n=top_n)

        # 风险流矩阵
        lines.append('  风险流矩阵 F_morning / F_evening / F_net')
        _append_matrix_edges(lines, 'F_morning (去程 morning)', ad['F_morning'],
                             nodes, g, town_data, top_n=top_n)
        _append_matrix_edges(lines, 'F_evening (回程 evening)', ad['F_evening'],
                             nodes, g, town_data, top_n=top_n)
        _append_matrix_edges(lines, 'F_net     (morning+evening)', ad['F_net'],
                             nodes, g, town_data, top_n=top_n)
        _append_matrix_edges(lines, 'F_net_norm (归一化后)', ad['F_net_norm'],
                             nodes, g, town_data, top_n=top_n)

        # 风险来源归属（每个县列出贡献最大的若干来源县）
        top_sources = ad['top_sources']
        order = np.argsort(-n_in, kind='stable')[:top_n]
        lines.append(f'  风险来源归属（按净流入排序的前 {len(order)} 个县，'
                     f'每县列出前 {5} 个来源）')
        for rank, i in enumerate(order, 1):
            i = int(i)
            srcs = top_sources.get(i, [])
            lines.append(f"    #{rank:<3d} node={str(nodes[i]):<6s} "
                         f"inflow={n_in[i]:.6g}  outflow={n_out[i]:.6g}  "
                         f"{_node_name(g, town_data, nodes[i])}")
            if not srcs:
                lines.append('        (no positive inflow source)')
            for sj, val in srcs:
                lines.append(f"        <- node={str(nodes[sj]):<6s} {val:>15.6g}   "
                             f"{_node_name(g, town_data, nodes[sj])}")
        lines.append('')

    return '\n'.join(lines)


# ============================================================
# ComputeWorker — 主计算执行线程
# ============================================================

class ComputeWorker(QThread):
    """背景载入资料、建立网络、计算 EpiRank。"""
    progress = Signal(int, int)
    log_message = Signal(str)
    finished_ok = Signal(dict)
    finished_err = Signal(str)

    def __init__(self, data_dir, d, max_loops):
        super().__init__()
        self.data_dir = data_dir
        self.d = d
        self.max_loops = max_loops

    def run(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.data_dir)

            town_data = {}
            g = nx.DiGraph()

            self.log_message.emit("Loading basic town data (bs.xlsx)...")
            build_basic_table_of_towns(town_data)
            self.log_message.emit(f"  Loaded {len(town_data)} towns.")

            self.log_message.emit("Loading Flu reported cases (Flu.xlsx)...")
            build_flu_reported_cases(town_data)

            self.log_message.emit("Loading Enterovirus reported cases (ev.xlsx)...")
            build_ev_reported_cases(town_data)

            self.log_message.emit("Loading SARS reported cases (SARS.xlsx)...")
            build_sars_reported_cases(town_data)

            self.log_message.emit("Building commuting network (cn.xlsx)...")
            build_commuting_network(g, town_data)
            self.log_message.emit(f"  Network: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

            # 主计算
            self.log_message.emit(
                f"Computing Reverse Sequential-day EpiRank (d={self.d}, max_loops={self.max_loops})...")
            (epidemic_risk, iterations, CN_C,
             transient_metrics, attribution_data) = compute_epidemic_risk(
                g, town_data, self.d, self.max_loops,
                progress_callback=lambda cur, mx: self.progress.emit(cur, mx),
                record_trajectory=True,
                record_attribution=True,
            )
            self.log_message.emit(f"  Converged after {iterations} iterations.")
            if transient_metrics is not None:
                self.log_message.emit(
                    f"  M1: abs_spectral_gap={transient_metrics['spectral_gap']:.5f}, "
                    f"Kemeny(sum 1/(1-lr))={transient_metrics['kemeny_constant']:.2f}")
            if attribution_data is not None:
                self.log_message.emit(
                    f"  M2: closed-form residual = "
                    f"{attribution_data['closed_form_residual']:.3e}")

            # M1 / M2 指标明细
            self.log_message.emit(
                format_m1_m2_log(list(g.nodes()), g, town_data,
                                 transient_metrics, attribution_data))

            # 验证归一化
            er_sum = float(epidemic_risk.sum())
            self.log_message.emit(f"  ER sum = {er_sum:.10f} (should be 1.0)")

            # 比較用的网络指标
            self.log_message.emit("Computing PageRank, HITS...")
            page_rank = nx.pagerank(g, alpha=self.d)
            hub_rank, authority_rank = nx.hits(g, max_iter=1000)

            nodes = list(g.nodes())
            ER_rank = {seq_no: float(epidemic_risk[i, 0]) for i, seq_no in enumerate(nodes)}
            pop_rank = {seq_no: town_data[g.nodes[seq_no][KEY_DB_ID]][KEY_POPULATION] for seq_no in nodes}
            flu_case_rank = {seq_no: town_data[g.nodes[seq_no][KEY_DB_ID]].get(KEY_FLU_TOTAL_CASES, 0) for seq_no in nodes}
            ev_case_rank = {seq_no: town_data[g.nodes[seq_no][KEY_DB_ID]].get(KEY_EV_AVERAGE_CASES, 0) for seq_no in nodes}
            sars_case_rank = {seq_no: town_data[g.nodes[seq_no][KEY_DB_ID]].get(KEY_SARS_TOTAL_CASES, 0) for seq_no in nodes}

            er_values = np.array(list(ER_rank.values()))
            ER_tot = float(er_values.sum())
            ER_avg = float(er_values.mean())
            ER_std = float(er_values.std())

            # ── Stage 7: Compute evaluation summary ──
            evaluation_summary = []  # 用于汇总所有评估指标

            # 辅助函数：基于 head/tail breaks 计算 Recall, Precision, F1
            def compute_classification_metrics(index_dict, disease_dict):
                common_keys = sorted(set(index_dict.keys()) & set(disease_dict.keys()))
                if len(common_keys) < 3:
                    return float('nan'), float('nan'), float('nan')
                idx_vals = np.array([index_dict[k] for k in common_keys])
                dis_vals = np.array([disease_dict[k] for k in common_keys])
                idx_breaks = head_tail_breaks(idx_vals, 3)
                dis_breaks = head_tail_breaks(dis_vals, 3)
                if len(idx_breaks) < 3 or len(dis_breaks) < 3:
                    return float('nan'), float('nan'), float('nan')
                idx_labels = classify_by_breaks(idx_vals, idx_breaks)
                dis_labels = classify_by_breaks(dis_vals, dis_breaks)
                pred_core = {k for k, lab in zip(common_keys, idx_labels) if lab != 'NC'}
                actual_core = {k for k, lab in zip(common_keys, dis_labels) if lab != 'NC'}
                tp = len(pred_core & actual_core)
                fp = len(pred_core - actual_core)
                fn = len(actual_core - pred_core)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                return recall, precision, f1

            # 待评估的网络指标
            index_dicts = [
                ('EpiRank', ER_rank),
                ('PageRank', page_rank),
                ('HITS-Hub', hub_rank),
                ('HITS-Authority', authority_rank),
                ('Population', pop_rank),
            ]
            # 疾病数据
            disease_dicts = [
                ('Flu', flu_case_rank),
                ('EV', ev_case_rank),
                ('SARS', sars_case_rank),
            ]

            for idx_name, idx_dict in index_dicts:
                for dis_name, dis_dict in disease_dicts:
                    common_keys = set(idx_dict.keys()) & set(dis_dict.keys())
                    if len(common_keys) > 2:
                        sub1 = {k: idx_dict[k] for k in common_keys}
                        sub2 = {k: dis_dict[k] for k in common_keys}
                        pr, pp = get_pearson_cor(sub1, sub2)
                        sr, sp = get_spearman_cor(sub1, sub2)
                        kr, kp = get_kendalltau_cor(sub1, sub2)
                        # 计算分类指标
                        recall, precision, f1 = compute_classification_metrics(idx_dict, dis_dict)
                        evaluation_summary.append({
                            'Index': idx_name,
                            'Disease': dis_name,
                            'Pearson_r': pr,
                            'Spearman_rho': sr,
                            'Kendall_tau': kr,
                            'Recall': recall,
                            'Precision': precision,
                            'F1': f1,
                        })

            ER_sorted = sorted_map(ER_rank)
            table_data = []
            for seq_no, er_val in ER_sorted:
                db_ID = g.nodes[seq_no][KEY_DB_ID]
                td = town_data[db_ID]
                table_data.append({
                    'seq_no': seq_no,
                    'post_code': g.nodes[seq_no][KEY_POST_CODE],
                    'db_ID': db_ID,
                    'county': td[KEY_COUNTY],
                    'town': td[KEY_TOWN],
                    'population': td[KEY_POPULATION],
                    'area': round(td[KEY_AREA], 2),
                    'density': round(td[KEY_DENSITY], 2),
                    'ERV': round(er_val, 8),
                    'ERP': round(100.0 * er_val / ER_tot, 4) if ER_tot > 0 else 0,
                    'C_local': td[KEY_LOCAL_COMMUTER_TYPE1],
                    'C_out': td[KEY_OUT_COMMUTER_TYPE1],
                    'C_in': td[KEY_IN_COMMUTER_TYPE1],
                    'page_rank': round(page_rank.get(seq_no, 0), 8),
                    'hub': round(hub_rank.get(seq_no, 0), 8),
                    'authority': round(authority_rank.get(seq_no, 0), 8),
                    'flu_cases': td.get(KEY_FLU_TOTAL_CASES, 0),
                    'ev_cases': td.get(KEY_EV_AVERAGE_CASES, 0),
                    'sars_cases': td.get(KEY_SARS_TOTAL_CASES, 0),
                    'pos_x': td[KEY_POS_XY][0],
                    'pos_y': td[KEY_POS_XY][1],
                })

            npos = {seq_no: town_data[g.nodes[seq_no][KEY_DB_ID]][KEY_POS_XY] for seq_no in nodes}

            results = {
                'g': g,
                'town_data': town_data,
                'epidemic_risk': epidemic_risk,
                'table_data': table_data,
                'ER_rank': ER_rank,
                'ER_tot': ER_tot, 'ER_avg': ER_avg, 'ER_std': ER_std,
                'evaluation_summary': evaluation_summary,
                'page_rank': page_rank,
                'hub_rank': hub_rank,
                'authority_rank': authority_rank,
                'flu_case_rank': flu_case_rank,
                'ev_case_rank': ev_case_rank,
                'sars_case_rank': sars_case_rank,
                'pop_rank': pop_rank,
                'iterations': iterations,
                'CN_C': CN_C,
                'npos': npos,
                'd': self.d,
                'transient_metrics': transient_metrics,
                'attribution_data': attribution_data,
            }
            self.finished_ok.emit(results)

        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            os.chdir(old_cwd)


# ============================================================
# SensitivityWorker — 对 d 做敏感度分析
# ============================================================

class SensitivityWorker(QThread):
    """扫描 d 值，计算 EpiRank 與疾病资料的相关性。"""
    progress = Signal(int, int)
    log_message = Signal(str)
    finished_ok = Signal(dict)
    finished_err = Signal(str)

    def __init__(self, data_dir, max_loops):
        super().__init__()
        self.data_dir = data_dir
        self.max_loops = max_loops

    def run(self):
        old_cwd = os.getcwd()
        try:
            os.chdir(self.data_dir)

            self.log_message.emit("=== Sensitivity Analysis ===")
            self.log_message.emit("Loading data files...")

            town_data = {}
            g = nx.DiGraph()
            build_basic_table_of_towns(town_data)
            build_flu_reported_cases(town_data)
            build_ev_reported_cases(town_data)
            build_commuting_network(g, town_data)

            nodes = list(g.nodes())
            flu_case_rank = {s: town_data[g.nodes[s][KEY_DB_ID]].get(KEY_FLU_TOTAL_CASES, 0) for s in nodes}
            ev_case_rank  = {s: town_data[g.nodes[s][KEY_DB_ID]].get(KEY_EV_AVERAGE_CASES, 0) for s in nodes}

            d_values = np.round(np.arange(0.05, 1.025, 0.05), 2)
            total = len(d_values)

            flu_pearson  = np.zeros(len(d_values))
            flu_spearman = np.zeros(len(d_values))
            ev_pearson   = np.zeros(len(d_values))
            ev_spearman  = np.zeros(len(d_values))

            for j, dv in enumerate(d_values):
                er, iters, _, _, _ = compute_epidemic_risk(
                    g, town_data, float(dv), self.max_loops,
                    record_trajectory=False, record_attribution=False)
                er_vals = np.array(er).flatten()
                er_rank = {nodes[k]: er_vals[k] for k in range(len(nodes))}

                fp_r, _ = get_pearson_cor(er_rank, flu_case_rank)
                fs_r, _ = get_spearman_cor(er_rank, flu_case_rank)
                ep_r, _ = get_pearson_cor(er_rank, ev_case_rank)
                es_r, _ = get_spearman_cor(er_rank, ev_case_rank)

                flu_pearson[j]  = fp_r
                flu_spearman[j] = fs_r
                ev_pearson[j]   = ep_r
                ev_spearman[j]  = es_r

                self.progress.emit(j + 1, total)
                self.log_message.emit(f"  d={dv:.2f} completed ({j+1}/{total})")

            self.log_message.emit(f"\nSensitivity analysis complete ({total} d values).")
            self.finished_ok.emit({
                'flu_pearson': flu_pearson,
                'flu_spearman': flu_spearman,
                'ev_pearson': ev_pearson,
                'ev_spearman': ev_spearman,
                'd_values': d_values,
            })

        except Exception as e:
            import traceback
            self.finished_err.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            os.chdir(old_cwd)


# ============================================================
# 主窗口
# ============================================================

class EpiRankMainWindow(QMainWindow):
    """主应用程序窗口（16 个 Tab）。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EpiRank - Epidemic Risk Analysis System (Reverse Sequential-day)")
        self.setMinimumSize(1200, 800)

        self.results = None
        self.sensitivity_results = None
        self.worker = None
        self.data_dir = os.path.dirname(os.path.abspath(__file__))

        self._build_menu()
        self._build_ui()
        self._update_status("Ready. Set parameters and click 'Run EpiRank'.")

    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File (&F)")

        act_set_dir = QAction("Set Data Directory...", self)
        act_set_dir.triggered.connect(self._choose_data_dir)
        file_menu.addAction(act_set_dir)

        act_export = QAction("Export Results to Excel...", self)
        act_export.triggered.connect(self._export_excel)
        file_menu.addAction(act_export)

        act_save_fig = QAction("Save Current Chart...", self)
        act_save_fig.triggered.connect(self._save_figure)
        file_menu.addAction(act_save_fig)

        file_menu.addSeparator()
        act_quit = QAction("Quit (&Q)", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 顶部参数与控制
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        param_group = QGroupBox("EpiRank Parameters")
        param_form = QFormLayout()

        self.spin_d = QDoubleSpinBox()
        self.spin_d.setRange(0.01, 0.99)
        self.spin_d.setSingleStep(0.01)
        self.spin_d.setValue(0.95)
        self.spin_d.setDecimals(2)
        self.spin_d.setToolTip("Damping factor (d).")
        param_form.addRow("Damping Factor (d):", self.spin_d)

        self.spin_loops = QSpinBox()
        self.spin_loops.setRange(100, 50000)
        self.spin_loops.setSingleStep(100)
        self.spin_loops.setValue(5000)
        self.spin_loops.setToolTip("Maximum number of iterations.")
        param_form.addRow("Max Iterations:", self.spin_loops)

        param_group.setLayout(param_form)
        top_layout.addWidget(param_group)

        dir_group = QGroupBox("Data Directory")
        dir_layout = QVBoxLayout()
        self.lbl_data_dir = QLabel(self.data_dir)
        self.lbl_data_dir.setWordWrap(True)
        dir_layout.addWidget(self.lbl_data_dir)
        btn_dir = QPushButton("Change...")
        btn_dir.clicked.connect(self._choose_data_dir)
        dir_layout.addWidget(btn_dir)
        dir_group.setLayout(dir_layout)
        top_layout.addWidget(dir_group)

        ctrl_group = QGroupBox("Control")
        ctrl_layout = QVBoxLayout()
        self.btn_run = QPushButton("Run EpiRank")
        self.btn_run.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self._run_computation)
        ctrl_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        ctrl_layout.addWidget(self.progress_bar)

        self.btn_sensitivity = QPushButton("Sensitivity Analysis (by d)")
        self.btn_sensitivity.setStyleSheet("font-size: 13px; padding: 6px;")
        self.btn_sensitivity.setEnabled(False)
        self.btn_sensitivity.setToolTip("Sweep damping factor d.")
        self.btn_sensitivity.clicked.connect(self._run_sensitivity)
        ctrl_layout.addWidget(self.btn_sensitivity)

        ctrl_group.setLayout(ctrl_layout)
        top_layout.addWidget(ctrl_group)

        main_layout.addWidget(top_widget)

        # 结果分页
        self.tabs = QTabWidget()

        # Tab 0: 结果表
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        filter_bar = QHBoxLayout()
        filter_bar.addWidget(QLabel("筛选区域:"))
        self.combo_region = QComboBox()
        self.combo_region.addItems([
            "全台湾 353 乡镇市",
            "大台北 gTaipei（48 乡镇市）",
        ])
        self.combo_region.currentIndexChanged.connect(self._populate_table)
        filter_bar.addWidget(self.combo_region)
        filter_bar.addStretch()
        table_layout.addLayout(filter_bar)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)
        self.tabs.addTab(table_widget, "Results Table")

        # Tab 1: 网络圖
        self.fig_network = Figure(figsize=(5, 10), dpi=100, facecolor='white')
        self.canvas_network = FigureCanvas(self.fig_network)
        self.toolbar_network = NavigationToolbar(self.canvas_network, self)
        net_widget = QWidget()
        net_layout = QVBoxLayout(net_widget)
        net_layout.addWidget(self.toolbar_network)
        net_layout.addWidget(self.canvas_network)
        self.tabs.addTab(net_widget, "Network Map")

        # Tab 2: 核心分类表
        table1_scroll = QScrollArea()
        table1_scroll.setWidgetResizable(True)
        table1_inner = QWidget()
        self.table1_layout = QVBoxLayout(table1_inner)
        self.table1_layout.setContentsMargins(6, 6, 6, 6)
        self.table1_layout.setSpacing(8)
        self.table1_widget = QTableWidget()
        self.table1_widget.setAlternatingRowColors(True)
        self.table1_layout.addWidget(self.table1_widget)
        table1_scroll.setWidget(table1_inner)
        self.tabs.addTab(table1_scroll, "Core Classification")

        # Tab: Evaluation Summary
        eval_widget = QWidget()
        eval_layout = QVBoxLayout(eval_widget)
        self.eval_table = QTableWidget()
        self.eval_table.setAlternatingRowColors(True)
        eval_layout.addWidget(self.eval_table)
        self.tabs.addTab(eval_widget, "Evaluation Summary")

        # Tab 4: 通勤流
        self.fig_analysis = Figure(figsize=(14, 18), dpi=100, facecolor='white')
        self.canvas_analysis = FigureCanvas(self.fig_analysis)
        self.toolbar_analysis = NavigationToolbar(self.canvas_analysis, self)
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        analysis_layout.addWidget(self.toolbar_analysis)
        analysis_layout.addWidget(self.canvas_analysis)
        self.tabs.addTab(analysis_widget, "Commuter Flow")

        # Tab 5: 疾病频率分布
        self.fig_disease = Figure(figsize=(12, 10), dpi=100, facecolor='white')
        self.canvas_disease = FigureCanvas(self.fig_disease)
        self.toolbar_disease = NavigationToolbar(self.canvas_disease, self)
        disease_widget = QWidget()
        disease_layout = QVBoxLayout(disease_widget)
        disease_layout.addWidget(self.toolbar_disease)
        disease_layout.addWidget(self.canvas_disease)
        self.tabs.addTab(disease_widget, "Frequency Distributions")

        # Tab 6: EpiRank 频率分布
        self.fig_epirank_dist = Figure(figsize=(10, 10), dpi=100, facecolor='white')
        self.canvas_epirank_dist = FigureCanvas(self.fig_epirank_dist)
        self.toolbar_epirank_dist = NavigationToolbar(self.canvas_epirank_dist, self)
        epirank_dist_widget = QWidget()
        epirank_dist_layout = QVBoxLayout(epirank_dist_widget)
        epirank_dist_layout.addWidget(self.toolbar_epirank_dist)
        epirank_dist_layout.addWidget(self.canvas_epirank_dist)
        self.tabs.addTab(epirank_dist_widget, "Frequency Distribution")

        # Tab 7: EpiRank vs 疾病
        self.fig_epirank_vs = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        self.canvas_epirank_vs = FigureCanvas(self.fig_epirank_vs)
        self.toolbar_epirank_vs = NavigationToolbar(self.canvas_epirank_vs, self)
        epirank_vs_widget = QWidget()
        epirank_vs_layout = QVBoxLayout(epirank_vs_widget)
        epirank_vs_layout.addWidget(self.toolbar_epirank_vs)
        epirank_vs_layout.addWidget(self.canvas_epirank_vs)
        self.tabs.addTab(epirank_vs_widget, "EpiRank vs Disease")

        # Tab 8: 指标比較
        self.fig_index_comp = Figure(figsize=(16, 5), dpi=100, facecolor='white')
        self.canvas_index_comp = FigureCanvas(self.fig_index_comp)
        self.toolbar_index_comp = NavigationToolbar(self.canvas_index_comp, self)
        index_comp_widget = QWidget()
        index_comp_layout = QVBoxLayout(index_comp_widget)
        index_comp_layout.addWidget(self.toolbar_index_comp)
        index_comp_layout.addWidget(self.canvas_index_comp)
        self.tabs.addTab(index_comp_widget, "Index Comparison")

        # Tab 9: 疾病地图
        self.fig_disease_map = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        self.canvas_disease_map = FigureCanvas(self.fig_disease_map)
        self.toolbar_disease_map = NavigationToolbar(self.canvas_disease_map, self)
        disease_map_widget = QWidget()
        disease_map_layout = QVBoxLayout(disease_map_widget)
        disease_map_layout.addWidget(self.toolbar_disease_map)
        disease_map_layout.addWidget(self.canvas_disease_map)
        self.tabs.addTab(disease_map_widget, "Disease Map")

        # Tab 10: EpiRank 地图
        self.fig_epirank_map = Figure(figsize=(10, 10), dpi=100, facecolor='white')
        self.canvas_epirank_map = FigureCanvas(self.fig_epirank_map)
        self.toolbar_epirank_map = NavigationToolbar(self.canvas_epirank_map, self)
        epirank_map_widget = QWidget()
        epirank_map_layout = QVBoxLayout(epirank_map_widget)
        epirank_map_layout.addWidget(self.toolbar_epirank_map)
        epirank_map_layout.addWidget(self.canvas_epirank_map)
        self.tabs.addTab(epirank_map_widget, "EpiRank Map")

        # Tab 11: EpiRank vs 疾病地图
        self.fig_overlay_map = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        self.canvas_overlay_map = FigureCanvas(self.fig_overlay_map)
        self.toolbar_overlay_map = NavigationToolbar(self.canvas_overlay_map, self)
        overlay_map_widget = QWidget()
        overlay_map_layout = QVBoxLayout(overlay_map_widget)
        overlay_map_layout.addWidget(self.toolbar_overlay_map)
        overlay_map_layout.addWidget(self.canvas_overlay_map)
        self.tabs.addTab(overlay_map_widget, "EpiRank vs Disease Map")

        # Tab 12: Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 11))
        self.tabs.addTab(self.log_text, "Log")

        # Tab 13: Sensitivity Analysis
        self.fig_sensitivity = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        self.canvas_sensitivity = FigureCanvas(self.fig_sensitivity)
        self.toolbar_sensitivity = NavigationToolbar(self.canvas_sensitivity, self)
        sensitivity_widget = QWidget()
        sensitivity_layout = QVBoxLayout(sensitivity_widget)
        sensitivity_layout.addWidget(self.toolbar_sensitivity)
        sensitivity_layout.addWidget(self.canvas_sensitivity)
        self.tabs.addTab(sensitivity_widget, "Sensitivity Analysis")

        # Tab 14: Transient Dynamics（M1）
        self.fig_transient = Figure(figsize=(14, 10), dpi=100, facecolor='white')
        self.canvas_transient = FigureCanvas(self.fig_transient)
        self.toolbar_transient = NavigationToolbar(self.canvas_transient, self)
        transient_widget = QWidget()
        transient_layout = QVBoxLayout(transient_widget)
        transient_layout.addWidget(self.toolbar_transient)
        transient_layout.addWidget(self.canvas_transient)
        self.tabs.addTab(transient_widget, "Transient Dynamics")

        # Tab 15: Attribution（M2）
        self.fig_attribution = Figure(figsize=(14, 10), dpi=100, facecolor='white')
        self.canvas_attribution = FigureCanvas(self.fig_attribution)
        self.toolbar_attribution = NavigationToolbar(self.canvas_attribution, self)
        attribution_widget = QWidget()
        attribution_layout = QVBoxLayout(attribution_widget)
        attribution_layout.addWidget(self.toolbar_attribution)
        attribution_layout.addWidget(self.canvas_attribution)
        self.tabs.addTab(attribution_widget, "Attribution")

        main_layout.addWidget(self.tabs, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _choose_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Directory", self.data_dir)
        if d:
            self.data_dir = d
            self.lbl_data_dir.setText(d)

    def _update_status(self, msg):
        self.status_bar.showMessage(msg)

    def _log(self, msg):
        self.log_text.append(msg)

    def _run_computation(self):
        required = ['bs.xlsx', 'Flu.xlsx', 'ev.xlsx', 'SARS.xlsx', 'cn.xlsx']
        missing = [f for f in required if not os.path.isfile(os.path.join(self.data_dir, f))]
        if missing:
            QMessageBox.warning(self, "Missing Data Files",
                                f"The following files are missing in {self.data_dir}:\n\n" +
                                "\n".join(missing))
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self._log(f"Data directory: {self.data_dir}")
        self._log(f"Parameters: d={self.spin_d.value()}, max_loops={self.spin_loops.value()}")
        self._log(f"Mode: Reverse Sequential-day (evening → morning, external factor at day end)")

        self.worker = ComputeWorker(
            self.data_dir,
            self.spin_d.value(),
            self.spin_loops.value()
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._log)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()
        self._update_status("Computing...")

    def _on_progress(self, cur, mx):
        if mx > 0:
            self.progress_bar.setValue(int(100 * cur / mx))

    def _on_finished_ok(self, results):
        self.results = results
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self._log(f"\nDone! Converged in {results['iterations']} iterations.")
        self._log(f"ER Total={results['ER_tot']:.6f}, Mean={results['ER_avg']:.6f}, StdDev={results['ER_std']:.6f}")
        self._update_status(f"Completed. {len(results['table_data'])} towns, {results['iterations']} iterations.")

        _tab_tasks = [
            ("Results Table",          self._populate_table),
            ("Network Graph",          self._draw_network),
            ("Data Analysis",          self._draw_data_analysis),
            ("Disease Analysis",       self._draw_disease_analysis),
            ("EpiRank Distribution",   self._draw_epirank_distribution),
            ("EpiRank vs Disease",     self._draw_epirank_vs_disease),
            ("Index Comparison",       self._draw_index_comparison),
            ("Disease Map",            self._draw_disease_map),
            ("EpiRank Map",            self._draw_epirank_map),
            ("Overlay Map",            self._draw_overlay_map),
            ("Detail Table",           self._populate_table1),
            ("Evaluation Summary",    self._populate_evaluation_summary),
            ("Transient Dynamics",     self._draw_transient),
            ("Attribution",            self._draw_attribution),
        ]
        for tab_name, func in _tab_tasks:
            try:
                func()
            except Exception as exc:
                self._log(f"WARNING: Failed to populate tab '{tab_name}': {exc}")

        self.btn_sensitivity.setEnabled(True)
        self._auto_save_results()
        self.tabs.setCurrentIndex(0)

    def closeEvent(self, event):
        for worker_attr in ('worker', 'sensitivity_worker'):
            w = getattr(self, worker_attr, None)
            if w is not None and w.isRunning():
                self._log(f"Waiting for {worker_attr} to finish...")
                w.quit()
                w.wait(5000)
        event.accept()

    def _on_finished_err(self, err_msg):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(0)
        self._log(f"\nERROR:\n{err_msg}")
        self._update_status("Error during computation.")
        QMessageBox.critical(self, "Computation Error", str(err_msg)[:500])

    # ---- Tab 0: 结果表 ----
    def _populate_table(self):
        if self.results is None:
            return

        all_data = self.results['table_data']
        if self.combo_region.currentIndex() == 1:
            data = [item for item in all_data if item['db_ID'] in GTAIPEI_DB_IDS]
        else:
            data = all_data

        headers = ['Rank', 'County', 'Town', 'ERV', 'ERP (%)',
                    'Population', 'Area', 'Density',
                    'C.local', 'C.out', 'C.in',
                    'PageRank', 'Hub', 'Authority',
                    'Flu Cases', 'EV Cases', 'SARS Cases']

        self.table.setRowCount(len(data))
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        ER_avg = self.results['ER_avg']
        ER_std = self.results['ER_std']

        for row, item in enumerate(data):
            erv = item['ERV']
            if erv >= ER_avg + 2 * ER_std:
                bg = QColor(255, 100, 100)
            elif erv >= ER_avg + ER_std:
                bg = QColor(255, 180, 100)
            elif erv >= ER_avg:
                bg = QColor(255, 255, 150)
            elif erv >= ER_avg - ER_std:
                bg = QColor(200, 230, 200)
            else:
                bg = QColor(220, 220, 220)

            values = [
                row + 1, item['county'], item['town'],
                f"{erv:.8f}", f"{item['ERP']:.4f}%",
                f"{item['population']:.0f}", item['area'], item['density'],
                item['C_local'], item['C_out'], item['C_in'],
                f"{item['page_rank']:.8f}", f"{item['hub']:.8f}", f"{item['authority']:.8f}",
                item['flu_cases'], item['ev_cases'], item['sars_cases']
            ]
            for col, val in enumerate(values):
                cell = QTableWidgetItem(str(val))
                cell.setBackground(bg)
                if col >= 3:
                    cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, cell)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    # ---- Tab 1: 网络圖 ----
    def _draw_network(self):
        self.fig_network.clear()
        ax = self.fig_network.add_subplot(111)

        g = self.results['g']
        npos = self.results['npos']
        ER_rank = self.results['ER_rank']
        ER_avg = self.results['ER_avg']
        ER_std = self.results['ER_std']

        dg = nx.Graph(g)
        node_list = list(dg.nodes())
        ncolor = []
        nsize = []
        for seq_no in node_list:
            er = ER_rank.get(seq_no, 0)
            nsize.append(max(er * 10, 0.002) ** 2 * 50000)
            if er >= ER_avg + 2 * ER_std:
                ncolor.append('red')
            elif er >= ER_avg + ER_std:
                ncolor.append('darkorange')
            elif er >= ER_avg:
                ncolor.append('gold')
            elif er >= ER_avg - ER_std:
                ncolor.append('lightgreen')
            else:
                ncolor.append('lightgray')

        max_size = max(nsize) if nsize else 1
        nsize = [s / max_size * 300 + 5 for s in nsize]

        nx.draw_networkx_edges(dg, pos=npos, width=0.1, alpha=0.12,
                                edge_color='steelblue', ax=ax)
        nx.draw_networkx_nodes(dg, pos=npos, nodelist=node_list,
                                node_size=nsize, node_color=ncolor,
                                linewidths=0.3, edgecolors='gray', ax=ax)

        ax.set_title(f"台湾乡镇市通勤网络：d={self.results['d']}",
                      fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
        self.fig_network.tight_layout()
        self.canvas_network.draw()

    # ---- Tab 4: 通勤流分析 ----
    def _draw_data_analysis(self):
        from matplotlib.gridspec import GridSpec

        self.fig_analysis.clear()
        gs = GridSpec(4, 3, figure=self.fig_analysis,
                      hspace=0.35, wspace=0.35,
                      left=0.06, right=0.97, top=0.97, bottom=0.04)

        g = self.results['g']
        town_data = self.results['town_data']
        nodes = list(g.nodes())

        db_ids    = [g.nodes[n][KEY_DB_ID] for n in nodes]
        positions = [town_data[db][KEY_POS_XY] for db in db_ids]
        xs        = [p[0] for p in positions]
        ys        = [p[1] for p in positions]
        norm_dens = [town_data[db][KEY_NORMALIZED_DENSITY] for db in db_ids]

        local_flows_arr = np.array(
            [g[n][n][KEY_COMMUTER_TYPE1] if g.has_edge(n, n) else 0
             for n in nodes], dtype=float)

        total_in  = np.array([town_data[db][KEY_IN_COMMUTER_TYPE1]
                              for db in db_ids], dtype=float)
        total_out = np.array([town_data[db][KEY_OUT_COMMUTER_TYPE1]
                              for db in db_ids], dtype=float)

        inter_in  = total_in  - local_flows_arr
        inter_out = total_out - local_flows_arr

        in_degs  = np.array([g.in_degree(n)  - (1 if g.has_edge(n, n) else 0)
                             for n in nodes], dtype=float)
        out_degs = np.array([g.out_degree(n) - (1 if g.has_edge(n, n) else 0)
                             for n in nodes], dtype=float)

        ax_a = self.fig_analysis.add_subplot(gs[0:3, 0:2])
        urban_idx   = [i for i, nd in enumerate(norm_dens) if nd > 0.10]
        regular_idx = [i for i, nd in enumerate(norm_dens) if 0.01 < nd <= 0.10]
        rural_idx   = [i for i, nd in enumerate(norm_dens) if nd <= 0.01]

        ax_a.scatter([xs[i] for i in rural_idx],   [ys[i] for i in rural_idx],
                     s=4, c='green', alpha=0.6, label='rural')
        ax_a.scatter([xs[i] for i in regular_idx],  [ys[i] for i in regular_idx],
                     s=10, c='royalblue', alpha=0.7, label='regular')
        ax_a.scatter([xs[i] for i in urban_idx],    [ys[i] for i in urban_idx],
                     s=22, c='purple', alpha=0.8, label='urbanized')
        ax_a.set_aspect('equal')
        ax_a.set_title('(a)', fontsize=11)
        ax_a.legend(fontsize=8, loc='upper left', markerscale=1.5)
        ax_a.axis('off')

        ax_b = self.fig_analysis.add_subplot(gs[0, 2])
        ax_b.scatter(in_degs, out_degs, s=8, c='steelblue', alpha=0.6)
        lim = (max(in_degs.max(), out_degs.max()) + 5) if len(in_degs) > 0 else 10
        ax_b.plot([0, lim], [0, lim], 'gray', linewidth=0.8, alpha=0.6)
        ax_b.set_xlim(0, lim)
        ax_b.set_ylim(0, lim)
        ax_b.set_xlabel('in degree', fontsize=8)
        ax_b.set_ylabel('out degree', fontsize=8)
        ax_b.set_title('(b)', fontsize=9)
        ax_b.tick_params(labelsize=6)

        ax_c = self.fig_analysis.add_subplot(gs[1, 2])
        ax_c.scatter(inter_in / 1e4, inter_out / 1e4, s=8, c='steelblue', alpha=0.6)
        flow_lim = (max(inter_in.max(), inter_out.max()) / 1e4 * 1.1) if len(inter_in) > 0 else 1.0
        ax_c.plot([0, flow_lim], [0, flow_lim], 'gray', linewidth=0.8, alpha=0.6)
        ax_c.set_xlim(0, flow_lim)
        ax_c.set_ylim(0, flow_lim)
        ax_c.set_xlabel(r'in flow ($\times 10^4$)', fontsize=8)
        ax_c.set_ylabel(r'out flow ($\times 10^4$)', fontsize=8)
        ax_c.set_title('(c)', fontsize=9)
        ax_c.tick_params(labelsize=6)

        ax_d = self.fig_analysis.add_subplot(gs[2, 2])
        log_ratios = []
        for i in range(len(nodes)):
            if inter_out[i] > 0 and inter_in[i] > 0:
                log_ratios.append(np.log10(inter_in[i] / inter_out[i]))
            else:
                log_ratios.append(0.0)
        log_ratios_sorted = sorted(log_ratios, reverse=True)
        n_pull = sum(1 for v in log_ratios_sorted if v > 0)
        ax_d.bar(range(len(log_ratios_sorted)), log_ratios_sorted,
                 width=1.0, color='steelblue', alpha=0.8)
        ax_d.axvline(x=n_pull, color='blue', linewidth=1.2)
        ax_d.axhline(y=0, color='black', linewidth=0.5)
        ax_d.set_xlabel('township rank', fontsize=8)
        ax_d.set_ylabel('log10(in / out)', fontsize=8)
        ax_d.set_title('(d)', fontsize=9)
        ax_d.tick_params(labelsize=6)

        edge_dists = []
        for u, v, edata in g.edges(data=True):
            if u == v:
                continue
            ux, uy = town_data[g.nodes[u][KEY_DB_ID]][KEY_POS_XY]
            vx, vy = town_data[g.nodes[v][KEY_DB_ID]][KEY_POS_XY]
            dist_km = np.sqrt((ux - vx)**2 + (uy - vy)**2) / 1000.0
            commuters = edata[KEY_COMMUTER_TYPE1]
            edge_dists.append((dist_km, commuters))

        ax_e = self.fig_analysis.add_subplot(gs[3, 0])
        if edge_dists:
            dist_arr = np.array([d for d, _ in edge_dists])
            weight_arr = np.array([w for _, w in edge_dists])
            total_commuters = weight_arr.sum()
            bins = np.arange(0, dist_arr.max() + 1, 1)
            ax_e.hist(dist_arr, bins=bins, weights=weight_arr / 1e4,
                      color='steelblue', edgecolor='white', linewidth=0.3, alpha=0.8)
            ax_e.set_xlabel('commuting distance (km)', fontsize=8)
            ax_e.set_ylabel(r'no. commuters ($\times 10^4$)', fontsize=8)
            ax_e.set_title('(e)', fontsize=9)
            ax_e.tick_params(labelsize=6)

        ax_f = self.fig_analysis.add_subplot(gs[3, 1])
        if edge_dists:
            sorted_edges = sorted(edge_dists, key=lambda x: x[0])
            cum_commuters = 0
            cum_x, cum_y = [0], [0]
            for d_km, w in sorted_edges:
                cum_commuters += w
                cum_x.append(d_km)
                cum_y.append(100.0 * cum_commuters / total_commuters)
            ax_f.plot(cum_x, cum_y, 'steelblue', linewidth=1.5)
            ax_f.set_xlim(0, max(cum_x))
            ax_f.set_ylim(0, 105)
            for pct in [80, 90]:
                ax_f.axhline(y=pct, color='gray', linestyle=':', linewidth=0.8)
                for cx, cy in zip(cum_x, cum_y):
                    if cy >= pct:
                        ax_f.axvline(x=cx, color='black', linestyle=':', linewidth=0.8)
                        ax_f.annotate(f'{pct}% -> {cx:.0f}km', xy=(cx, pct),
                                      fontsize=7, color='red',
                                      xytext=(cx + 1, pct - 5))
                        break
            ax_f.set_xlabel('commuting distance (km)', fontsize=8)
            ax_f.set_ylabel('cumulative %', fontsize=8)
            ax_f.set_title('(f)', fontsize=9)
            ax_f.tick_params(labelsize=6)

        ax_g = self.fig_analysis.add_subplot(gs[3, 2])
        if local_flows_arr.max() > 0:
            bins_g = np.linspace(0, local_flows_arr.max() * 1.05, 11)
            ax_g.hist(local_flows_arr, bins=bins_g, color='#d2691e',
                      edgecolor='white', linewidth=0.3, alpha=0.8)
            ax_g.set_xlabel('local flow', fontsize=8)
            ax_g.set_ylabel('no. townships', fontsize=8)
            ax_g.set_title('(g)', fontsize=9)
            ax_g.tick_params(labelsize=6)

        self.canvas_analysis.draw()

    # ---- Tab 5: 疾病频率分布 ----
    def _draw_disease_analysis(self):
        from matplotlib.gridspec import GridSpec

        self.fig_disease.clear()
        gs = GridSpec(2, 2, figure=self.fig_disease,
                      hspace=0.38, wspace=0.30,
                      left=0.08, right=0.96, top=0.94, bottom=0.08)

        g = self.results['g']
        town_data = self.results['town_data']
        nodes = list(g.nodes())
        db_ids = [g.nodes[n][KEY_DB_ID] for n in nodes]

        flu_cases = np.array([town_data[db].get(KEY_FLU_TOTAL_CASES, 0)
                              for db in db_ids], dtype=float)
        ev_cases  = np.array([town_data[db].get(KEY_EV_AVERAGE_CASES, 0.0)
                              for db in db_ids], dtype=float)

        local_flows_arr = np.array(
            [g[n][n][KEY_COMMUTER_TYPE1] if g.has_edge(n, n) else 0
             for n in nodes], dtype=float)
        total_in  = np.array([town_data[db][KEY_IN_COMMUTER_TYPE1]
                              for db in db_ids], dtype=float)
        total_out = np.array([town_data[db][KEY_OUT_COMMUTER_TYPE1]
                              for db in db_ids], dtype=float)
        inter_in  = total_in  - local_flows_arr
        inter_out = total_out - local_flows_arr

        log_ratios = np.zeros(len(nodes))
        for i in range(len(nodes)):
            if inter_out[i] > 0 and inter_in[i] > 0:
                log_ratios[i] = np.log10(inter_in[i] / inter_out[i])

        flu_breaks = head_tail_breaks(flu_cases, 3)
        ev_breaks  = head_tail_breaks(ev_cases, 3)
        flu_levels = classify_by_breaks(flu_cases, flu_breaks)
        ev_levels  = classify_by_breaks(ev_cases, ev_breaks)

        def draw_freq_dist(ax, case_values, breaks, levels, title,
                           xlabel='epidemic risk'):
            vmin = max(0, case_values.min())
            vmax = case_values.max()
            if vmax <= 50 and np.allclose(case_values, case_values.astype(int)):
                bins = np.arange(vmin, vmax + 2, 1) - 0.5
            else:
                bins = np.linspace(vmin - 0.25, vmax + 0.25, 35)

            level_data = {lv: [] for lv in LEVEL_ORDER}
            for val, lv in zip(case_values, levels):
                level_data[lv].append(val)

            bottom = np.zeros(len(bins) - 1)
            for lv in LEVEL_ORDER:
                if not level_data[lv]:
                    continue
                counts, _ = np.histogram(level_data[lv], bins=bins)
                ax.bar(bins[:-1] + np.diff(bins) / 2, counts,
                       width=np.diff(bins) * 0.92,
                       bottom=bottom, color=LEVEL_COLORS[lv],
                       edgecolor='white', linewidth=0.3, label=lv)
                bottom += counts

            for bk in breaks:
                ax.axvline(x=bk, color='black', linestyle='--', linewidth=0.9)

            ylim_top = ax.get_ylim()[1]
            edges = [vmin] + list(breaks) + [vmax]
            for i, lv in enumerate(LEVEL_ORDER):
                if i < len(edges) - 1:
                    mid_x = (edges[i] + edges[i + 1]) / 2
                    ax.text(mid_x, ylim_top * 0.97, lv, ha='center',
                            va='top', fontsize=8, fontweight='bold',
                            color='blue')

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('number of townships', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)

        def draw_inout_ratio(ax, log_ratios, levels, title):
            bins = np.linspace(-1.0, 1.0, 11)
            level_data = {lv: [] for lv in LEVEL_ORDER}
            for lr, lv in zip(log_ratios, levels):
                level_data[lv].append(lr)

            bottom = np.zeros(len(bins) - 1)
            for lv in LEVEL_ORDER:
                if not level_data[lv]:
                    continue
                counts, _ = np.histogram(level_data[lv], bins=bins)
                ax.bar(bins[:-1] + np.diff(bins) / 2, counts,
                       width=np.diff(bins) * 0.92,
                       bottom=bottom, color=LEVEL_COLORS[lv],
                       edgecolor='white', linewidth=0.3, label=lv)
                bottom += counts

            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.0)

            ylim = ax.get_ylim()
            y_label = ylim[1] * 0.95 if ylim[1] > 0 else 1
            ax.text(-0.95, y_label, 'push', ha='left', va='top',
                    fontsize=9, fontweight='bold', color='#333333')
            ax.text(0.95, y_label, 'pull', ha='right', va='top',
                    fontsize=9, fontweight='bold', color='#333333')

            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel(r'$log_{10}$(in/out)', fontsize=9)
            ax.set_ylabel('number of townships', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)

        ax_a = self.fig_disease.add_subplot(gs[0, 0])
        draw_freq_dist(ax_a, flu_cases, flu_breaks, flu_levels,
                       '(a) frequency distribution of flu cases')

        ax_b = self.fig_disease.add_subplot(gs[0, 1])
        draw_freq_dist(ax_b, ev_cases, ev_breaks, ev_levels,
                       '(b) frequency distribution of EV cases')

        ax_c = self.fig_disease.add_subplot(gs[1, 0])
        draw_inout_ratio(ax_c, log_ratios, flu_levels,
                         '(c) in/out ratio of flu cases')

        ax_d = self.fig_disease.add_subplot(gs[1, 1])
        draw_inout_ratio(ax_d, log_ratios, ev_levels,
                         '(d) in/out ratio of EV cases')

        self.canvas_disease.draw()

    # ---- Tab 6: EpiRank 频率分布（1×2）----
    def _draw_epirank_distribution(self):
        from matplotlib.gridspec import GridSpec

        self.fig_epirank_dist.clear()
        gs = GridSpec(2, 1, figure=self.fig_epirank_dist,
                      hspace=0.42,
                      left=0.12, right=0.95, top=0.92, bottom=0.08)

        g = self.results['g']
        town_data = self.results['town_data']
        nodes = list(g.nodes())
        db_ids = [g.nodes[n][KEY_DB_ID] for n in nodes]

        er_matrix = self.results['epidemic_risk']
        er_values = np.array([float(er_matrix[i, 0]) for i in range(len(nodes))])

        local_flows_arr = np.array(
            [g[n][n][KEY_COMMUTER_TYPE1] if g.has_edge(n, n) else 0
             for n in nodes], dtype=float)
        total_in = np.array([town_data[db][KEY_IN_COMMUTER_TYPE1]
                             for db in db_ids], dtype=float)
        total_out = np.array([town_data[db][KEY_OUT_COMMUTER_TYPE1]
                              for db in db_ids], dtype=float)
        inter_in = total_in - local_flows_arr
        inter_out = total_out - local_flows_arr
        log_ratios = np.zeros(len(nodes))
        for i in range(len(nodes)):
            if inter_out[i] > 0 and inter_in[i] > 0:
                log_ratios[i] = np.log10(inter_in[i] / inter_out[i])

        er_breaks = head_tail_breaks(er_values, 3)
        er_levels = classify_by_breaks(er_values, er_breaks)

        # (a) EpiRank 频率分布
        ax_top = self.fig_epirank_dist.add_subplot(gs[0, 0])
        bins = np.linspace(0, max(er_values.max() * 1.05, 0.016), 30)

        level_data = {lv: [] for lv in LEVEL_ORDER}
        for val, lv in zip(er_values, er_levels):
            level_data[lv].append(val)

        bottom = np.zeros(len(bins) - 1)
        for lv in LEVEL_ORDER:
            if not level_data[lv]:
                continue
            counts, _ = np.histogram(level_data[lv], bins=bins)
            ax_top.bar(bins[:-1] + np.diff(bins) / 2, counts,
                       width=np.diff(bins) * 0.92,
                       bottom=bottom, color=LEVEL_COLORS[lv],
                       edgecolor='white', linewidth=0.3, label=lv)
            bottom += counts

        for bk in er_breaks:
            ax_top.axvline(x=bk, color='black', linestyle='--', linewidth=0.9)

        ylim_top = ax_top.get_ylim()[1]
        edges = [0] + list(er_breaks) + [er_values.max() * 1.1]
        abbrev_labels = ['NC', 'III', 'II', 'I']
        for i_lv, lv_abbrev in enumerate(abbrev_labels):
            if i_lv < len(edges) - 1:
                mid_x = (edges[i_lv] + edges[i_lv + 1]) / 2
                ax_top.text(mid_x, ylim_top * 0.97, lv_abbrev, ha='center',
                            va='top', fontsize=8, fontweight='bold', color='blue')

        ax_top.annotate('Core', xy=(er_breaks[-1], ylim_top * 0.5),
                        xytext=(er_values.max() * 0.85, ylim_top * 0.7),
                        fontsize=9, fontweight='bold', color='#555555',
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

        ax_top.set_xlabel('EpiRank', fontsize=9)
        ax_top.set_ylabel('number of townships', fontsize=9)
        ax_top.set_title('(a) EpiRank frequency distribution', fontsize=10)
        ax_top.tick_params(labelsize=7)

        # (b) log in/out ratio
        ax_bot = self.fig_epirank_dist.add_subplot(gs[1, 0])
        ratio_bins = np.linspace(-1.0, 1.0, 11)
        ratio_level_data = {lv: [] for lv in LEVEL_ORDER}
        for lr, lv in zip(log_ratios, er_levels):
            ratio_level_data[lv].append(lr)

        bottom = np.zeros(len(ratio_bins) - 1)
        for lv in LEVEL_ORDER:
            if not ratio_level_data[lv]:
                continue
            counts, _ = np.histogram(ratio_level_data[lv], bins=ratio_bins)
            ax_bot.bar(ratio_bins[:-1] + np.diff(ratio_bins) / 2, counts,
                       width=np.diff(ratio_bins) * 0.92,
                       bottom=bottom, color=LEVEL_COLORS[lv],
                       edgecolor='white', linewidth=0.3, label=lv)
            bottom += counts

        ax_bot.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
        ylim_b = ax_bot.get_ylim()
        y_label = ylim_b[1] * 0.95 if ylim_b[1] > 0 else 1
        ax_bot.text(-0.95, y_label, 'push', ha='left', va='top',
                    fontsize=9, fontweight='bold', color='#333333')
        ax_bot.text(0.95, y_label, 'pull', ha='right', va='top',
                    fontsize=9, fontweight='bold', color='#333333')

        ax_bot.set_xlim(-1.0, 1.0)
        ax_bot.set_xlabel(r'$log_{10}$(in/out)', fontsize=9)
        ax_bot.set_ylabel('number of townships', fontsize=9)
        ax_bot.set_title('(b) log in/out ratio by EpiRank level', fontsize=10)
        ax_bot.tick_params(labelsize=7)

        self.canvas_epirank_dist.draw()

    # ---- Tab 7: EpiRank vs 疾病 ----
    def _draw_epirank_vs_disease(self):
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Patch

        self.fig_epirank_vs.clear()
        gs = GridSpec(1, 2, figure=self.fig_epirank_vs,
                      wspace=0.30,
                      left=0.08, right=0.95, top=0.88, bottom=0.12)

        g = self.results['g']
        town_data = self.results['town_data']
        nodes = list(g.nodes())
        db_ids = [g.nodes[n][KEY_DB_ID] for n in nodes]

        er_matrix = self.results['epidemic_risk']
        er_values = np.array([float(er_matrix[i, 0]) for i in range(len(nodes))])
        er_breaks = head_tail_breaks(er_values, 3)
        er_levels = classify_by_breaks(er_values, er_breaks)

        flu_cases = np.array([town_data[db].get(KEY_FLU_TOTAL_CASES, 0)
                              for db in db_ids], dtype=float)
        ev_cases = np.array([town_data[db].get(KEY_EV_AVERAGE_CASES, 0.0)
                             for db in db_ids], dtype=float)

        flu_breaks = head_tail_breaks(flu_cases, 3)
        ev_breaks = head_tail_breaks(ev_cases, 3)
        flu_actual = classify_by_breaks(flu_cases, flu_breaks)
        ev_actual = classify_by_breaks(ev_cases, ev_breaks)

        X_LEVELS = ['core-I', 'core-II', 'core-III', 'non-core']
        LEVEL_TO_X = {'C-I': 'core-I', 'C-II': 'core-II',
                      'C-III': 'core-III', 'NC': 'non-core'}
        PREDICTED_COLORS = {
            'core-I':    '#cc2020',
            'core-II':   '#e07830',
            'core-III':  '#c8b840',
            'non-core':  '#3a8f3e',
        }

        def draw_comparison(ax, actual_levels, predicted_levels, title):
            actual_x = [LEVEL_TO_X[a] for a in actual_levels]
            predicted_x = [LEVEL_TO_X[p] for p in predicted_levels]

            group_counts = {}
            group_totals = {}
            for a, p in zip(actual_x, predicted_x):
                if a not in group_counts:
                    group_counts[a] = {xl: 0 for xl in X_LEVELS}
                    group_totals[a] = 0
                group_counts[a][p] += 1
                group_totals[a] += 1

            x_positions = np.arange(len(X_LEVELS))
            bar_width = 0.6

            bottom = np.zeros(len(X_LEVELS))
            for pred_lv in X_LEVELS:
                pcts = []
                for act_lv in X_LEVELS:
                    total = group_totals.get(act_lv, 0)
                    count = group_counts.get(act_lv, {}).get(pred_lv, 0)
                    pct = (100.0 * count / total) if total > 0 else 0
                    pcts.append(pct)

                ax.bar(x_positions, pcts, bar_width,
                       bottom=bottom,
                       color=PREDICTED_COLORS[pred_lv],
                       edgecolor='white', linewidth=0.5,
                       label=pred_lv)
                bottom += np.array(pcts)

            for i, act_lv in enumerate(X_LEVELS):
                total = group_totals.get(act_lv, 0)
                ax.text(i, 103, str(total), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

            ax.set_xticks(x_positions)
            ax.set_xticklabels(X_LEVELS, fontsize=9)
            ax.set_xlabel('actual condition', fontsize=10)
            ax.set_ylabel('percentage', fontsize=10)
            ax.set_ylim(0, 115)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_yticklabels(['0 %', '20 %', '40 %', '60 %', '80 %', '100 %'],
                               fontsize=8)
            ax.set_title(title, fontsize=11)
            ax.tick_params(labelsize=8)

        ax_a = self.fig_epirank_vs.add_subplot(gs[0, 0])
        draw_comparison(ax_a, flu_actual, er_levels, '(a) flu case')

        ax_b = self.fig_epirank_vs.add_subplot(gs[0, 1])
        draw_comparison(ax_b, ev_actual, er_levels, '(b) EV case')

        legend_patches = [
            Patch(facecolor=PREDICTED_COLORS[lv], label=lv)
            for lv in X_LEVELS
        ]
        self.fig_epirank_vs.legend(
            handles=legend_patches, loc='upper right',
            title='predicted condition', fontsize=8, title_fontsize=9,
            frameon=True, bbox_to_anchor=(0.98, 0.98))

        self.canvas_epirank_vs.draw()

    # ---- Tab 2: 核心分类表 ----
    def _populate_table1(self):
        if self.results is None:
            return

        nodes = list(self.results['g'].nodes())

        # === Table 1: Core counts by index method ===
        metrics = [
            ('EpiRank', self.results['ER_rank']),
            ('PageRank', self.results['page_rank']),
            ('HITS-Hub', self.results['hub_rank']),
            ('HITS-Authority', self.results['authority_rank']),
        ]

        row_labels = ['core-I', 'core-II', 'core-III', 'non-core']
        level_to_row = {'C-I': 0, 'C-II': 1, 'C-III': 2, 'NC': 3}
        row_colors = [
            QColor(255, 120, 120),
            QColor(255, 190, 120),
            QColor(255, 255, 150),
            QColor(180, 230, 180),
        ]

        self.table1_widget.setRowCount(len(row_labels))
        self.table1_widget.setColumnCount(len(metrics) + 1)
        self.table1_widget.setHorizontalHeaderLabels([''] + [m[0] for m in metrics])
        self.table1_widget.setVerticalHeaderLabels([])

        for r, label in enumerate(row_labels):
            item = QTableWidgetItem(label)
            item.setFont(QFont('Arial', 11, QFont.Bold))
            item.setBackground(row_colors[r])
            self.table1_widget.setItem(r, 0, item)

        for col_idx, (metric_name, metric_dict) in enumerate(metrics):
            values = np.array([metric_dict.get(n, 0) for n in nodes])
            breaks = head_tail_breaks(values, 3)
            levels = classify_by_breaks(values, breaks)

            counts = {'C-I': 0, 'C-II': 0, 'C-III': 0, 'NC': 0}
            for lv in levels:
                counts[lv] += 1

            for lv, count in counts.items():
                r = level_to_row[lv]
                item = QTableWidgetItem(str(count))
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont('Arial', 12))
                item.setBackground(row_colors[r])
                self.table1_widget.setItem(r, col_idx + 1, item)

        self.table1_widget.resizeColumnsToContents()
        self.table1_widget.horizontalHeader().setStretchLastSection(True)
        for c in range(self.table1_widget.columnCount()):
            self.table1_widget.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)

        # === Remove previously added overlap tables ===
        while self.table1_layout.count() > 1:
            item = self.table1_layout.takeAt(self.table1_layout.count() - 1)
            w = item.widget()
            if w:
                w.deleteLater()

        # === Spatial Overlap tables: EpiRank vs each disease ===
        core_labels = self._compute_core_labels_dict()
        epirank_labels = core_labels.get('EpiRank', {})

        overlap_headers = ['|EpiRank|', '|Disease|', '|A∩B|', '|A∪B|',
                           'IoU', 'Dice', 'Cov.(EpiRank)', 'Cov.(Disease)']

        overlap_row_colors = [
            QColor(255, 120, 120),
            QColor(255, 190, 120),
            QColor(255, 255, 150),
            QColor(180, 230, 180),
            QColor(200, 200, 200),
        ]

        diseases = [
            ('Flu',  core_labels.get('Flu', {})),
            ('EV',   core_labels.get('EV', {})),
            ('SARS', core_labels.get('SARS', {})),
        ]

        for disease_name, dis_labels in diseases:
            rows_data = []
            for lvl_name, lvl_set in OVERLAP_LEVEL_SETS:
                res = compute_overlap_metrics(epirank_labels, dis_labels, lvl_set)
                rows_data.append((lvl_name, res))

            num_data_rows = len(rows_data)
            num_cols = len(overlap_headers) + 1
            tbl = QTableWidget(num_data_rows + 1, num_cols)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.setAlternatingRowColors(True)

            title_item = QTableWidgetItem(f'EpiRank({disease_name})')
            title_item.setFont(QFont('Arial', 11, QFont.Bold))
            title_item.setBackground(QColor(180, 210, 230))
            title_item.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(0, 0, title_item)

            for c, h in enumerate(overlap_headers):
                item = QTableWidgetItem(h)
                item.setFont(QFont('Arial', 9, QFont.Bold))
                item.setBackground(QColor(230, 230, 230))
                item.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(0, c + 1, item)

            for r, (lvl_name, res) in enumerate(rows_data):
                item = QTableWidgetItem(lvl_name)
                item.setFont(QFont('Arial', 10, QFont.Bold))
                item.setBackground(overlap_row_colors[r])
                tbl.setItem(r + 1, 0, item)

                iou = res['iou']
                if iou >= 0.9:
                    bg = QColor(200, 240, 200)
                elif iou >= 0.6:
                    bg = QColor(255, 240, 180)
                else:
                    bg = QColor(255, 210, 210)

                vals = [
                    str(res['n_a']), str(res['n_b']),
                    str(res['n_i']), str(res['n_u']),
                    f"{res['iou']:.4f}", f"{res['dice']:.4f}",
                    f"{res['cov_a']:.4f}", f"{res['cov_b']:.4f}",
                ]
                for c, v in enumerate(vals):
                    cell = QTableWidgetItem(v)
                    cell.setTextAlignment(Qt.AlignCenter)
                    if c >= 4:
                        cell.setBackground(bg)
                    tbl.setItem(r + 1, c + 1, cell)

            for c in range(tbl.columnCount()):
                tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)
            self.table1_layout.addWidget(tbl)

        self.table1_layout.addStretch()

    # ---- Tab 8: 指标比較 ----
    def _draw_index_comparison(self):
        from matplotlib.gridspec import GridSpec

        self.fig_index_comp.clear()
        gs = GridSpec(1, 4, figure=self.fig_index_comp,
                      wspace=0.30,
                      left=0.05, right=0.97, top=0.88, bottom=0.14)

        g = self.results['g']
        town_data = self.results['town_data']
        nodes = list(g.nodes())
        db_ids = [g.nodes[n][KEY_DB_ID] for n in nodes]

        local_flows_arr = np.array(
            [g[n][n][KEY_COMMUTER_TYPE1] if g.has_edge(n, n) else 0
             for n in nodes], dtype=float)
        total_in = np.array([town_data[db][KEY_IN_COMMUTER_TYPE1]
                             for db in db_ids], dtype=float)
        total_out = np.array([town_data[db][KEY_OUT_COMMUTER_TYPE1]
                              for db in db_ids], dtype=float)
        inter_in = total_in - local_flows_arr
        inter_out = total_out - local_flows_arr

        log_ratios = np.zeros(len(nodes))
        for i in range(len(nodes)):
            if inter_out[i] > 0 and inter_in[i] > 0:
                log_ratios[i] = np.log10(inter_in[i] / inter_out[i])

        metrics = [
            ('(a) EpiRank', self.results['ER_rank']),
            ('(b) PageRank', self.results['page_rank']),
            ('(c) HITS-Hub', self.results['hub_rank']),
            ('(d) HITS-Authority', self.results['authority_rank']),
        ]

        COLOR_CORE = '#cc2020'
        COLOR_NONCORE = '#3a8f3e'

        for col_idx, (title, metric_dict) in enumerate(metrics):
            ax = self.fig_index_comp.add_subplot(gs[0, col_idx])

            values = np.array([metric_dict.get(n, 0) for n in nodes])
            breaks = head_tail_breaks(values, 3)
            levels = classify_by_breaks(values, breaks)

            core_ratios = [lr for lr, lv in zip(log_ratios, levels) if lv != 'NC']
            noncore_ratios = [lr for lr, lv in zip(log_ratios, levels) if lv == 'NC']

            bins = np.linspace(-1.0, 1.0, 11)
            nc_counts, _ = np.histogram(noncore_ratios, bins=bins)
            c_counts, _ = np.histogram(core_ratios, bins=bins)
            bar_centers = bins[:-1] + np.diff(bins) / 2
            bar_width = np.diff(bins) * 0.92

            ax.bar(bar_centers, nc_counts, width=bar_width,
                   color=COLOR_NONCORE, edgecolor='white', linewidth=0.3,
                   label='non-core')
            ax.bar(bar_centers, c_counts, width=bar_width,
                   bottom=nc_counts, color=COLOR_CORE,
                   edgecolor='white', linewidth=0.3,
                   label='core')

            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
            ax.set_xlim(-1.0, 1.0)
            max_height = max((nc_counts + c_counts).max(), 1)
            y_upper = int(np.ceil(max_height * 1.1 / 10)) * 10
            ax.set_ylim(0, y_upper)
            ax.text(-0.95, y_upper * 0.95, 'push', ha='left', va='top',
                    fontsize=9, fontweight='bold', color='#333333')
            ax.text(0.95, y_upper * 0.95, 'pull', ha='right', va='top',
                    fontsize=9, fontweight='bold', color='#333333')

            ax.set_xlabel(r'$log_{10}$(in/out)', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('number of townships', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=COLOR_CORE, label='core'),
            Patch(facecolor=COLOR_NONCORE, label='non-core'),
        ]
        self.fig_index_comp.legend(handles=legend_handles,
                                   loc='upper right', fontsize=9,
                                   frameon=True, bbox_to_anchor=(0.99, 0.99))

        self.canvas_index_comp.draw()

    # ---- Tab 9: 疾病地图 ----
    def _draw_disease_map(self):
        self.fig_disease_map.clear()

        g = self.results['g']
        npos = self.results['npos']

        disease_info = [
            ('(a) flu case distribution', self.results['flu_case_rank']),
            ('(b) EV case distribution', self.results['ev_case_rank']),
        ]
        size_map = {'C-I': 120, 'C-II': 60, 'C-III': 30, 'NC': 10}
        legend_labels = {'NC': 'non-core', 'C-III': 'core-III',
                         'C-II': 'core-II', 'C-I': 'core-I'}

        for col, (title, case_dict) in enumerate(disease_info):
            ax = self.fig_disease_map.add_subplot(1, 2, col + 1)
            vals = np.array(list(case_dict.values()))
            keys = list(case_dict.keys())
            breaks = head_tail_breaks(vals)
            if len(breaks) < 3:
                ax.set_title(title + '\n(insufficient data)')
                continue
            labels = classify_by_breaks(vals, breaks)

            for level in LEVEL_ORDER:
                xs = [npos[k][0] for k, lab in zip(keys, labels) if lab == level and k in npos]
                ys = [npos[k][1] for k, lab in zip(keys, labels) if lab == level and k in npos]
                if xs:
                    marker = '+' if level == 'NC' else 'o'
                    kw = dict(s=size_map[level], c=LEVEL_COLORS[level],
                              marker=marker, linewidths=0.3,
                              label=legend_labels[level],
                              zorder=2 + LEVEL_ORDER.index(level))
                    if level != 'NC':
                        kw['edgecolors'] = 'gray'
                    ax.scatter(xs, ys, **kw)

            ax.set_title(title, fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend(loc='lower right', fontsize=8, framealpha=0.8,
                      title='disease severity level', title_fontsize=8)

        self.fig_disease_map.tight_layout()
        self.canvas_disease_map.draw()

    # ---- Tab 10: EpiRank 地图（2×1）----
    def _draw_epirank_map(self):
        self.fig_epirank_map.clear()

        g = self.results['g']
        npos = self.results['npos']
        nodes_list = list(g.nodes())

        er_matrix = self.results['epidemic_risk']
        er_vals = np.array(er_matrix).flatten()
        er_rank = {nodes_list[i]: er_vals[i] for i in range(len(nodes_list))}

        vals_arr = np.array(list(er_rank.values()))
        keys_list = list(er_rank.keys())
        breaks = head_tail_breaks(vals_arr)
        labels = classify_by_breaks(vals_arr, breaks)
        label_map = {k: lab for k, lab in zip(keys_list, labels)}

        size_map_tw = {'C-I': 40, 'C-II': 20, 'C-III': 10, 'NC': 3}
        size_map_tp = {'C-I': 80, 'C-II': 40, 'C-III': 20, 'NC': 8}
        legend_labels = {'NC': 'non-core', 'C-III': 'core-III',
                         'C-II': 'core-II', 'C-I': 'core-I'}

        gs = self.fig_epirank_map.add_gridspec(2, 1, hspace=0.15, top=0.92)

        taipei_nodes = [s for s in nodes_list if g.nodes[s][KEY_DB_ID] in GTAIPEI_DB_IDS]

        # 上：全台湾
        ax0 = self.fig_epirank_map.add_subplot(gs[0, 0])
        for level in LEVEL_ORDER:
            xs = [npos[k][0] for k in keys_list if label_map[k] == level and k in npos]
            ys = [npos[k][1] for k in keys_list if label_map[k] == level and k in npos]
            if xs:
                ax0.scatter(xs, ys, s=size_map_tw[level], c=LEVEL_COLORS[level],
                            edgecolors='gray', linewidths=0.3,
                            label=legend_labels[level],
                            zorder=2 + LEVEL_ORDER.index(level))
        ax0.set_title('(a) EpiRank — Taiwan', fontsize=10)
        ax0.set_aspect('equal')
        ax0.axis('off')

        # 下：大台北
        ax1 = self.fig_epirank_map.add_subplot(gs[1, 0])
        for level in LEVEL_ORDER:
            xs = [npos[k][0] for k in taipei_nodes if label_map.get(k) == level and k in npos]
            ys = [npos[k][1] for k in taipei_nodes if label_map.get(k) == level and k in npos]
            if xs:
                ax1.scatter(xs, ys, s=size_map_tp[level], c=LEVEL_COLORS[level],
                            edgecolors='gray', linewidths=0.3,
                            label=legend_labels[level],
                            zorder=2 + LEVEL_ORDER.index(level))
        if taipei_nodes:
            tx = [npos[k][0] for k in taipei_nodes if k in npos]
            ty = [npos[k][1] for k in taipei_nodes if k in npos]
            if tx and ty:
                pad_x = (max(tx) - min(tx)) * 0.08 or 0.5
                pad_y = (max(ty) - min(ty)) * 0.08 or 0.5
                ax1.set_xlim(min(tx) - pad_x, max(tx) + pad_x)
                ax1.set_ylim(min(ty) - pad_y, max(ty) + pad_y)
        ax1.set_title('(b) EpiRank — Taipei Metropolitan Area', fontsize=10)
        ax1.set_aspect('equal')
        ax1.axis('off')

        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=LEVEL_COLORS[lv], label=legend_labels[lv])
                          for lv in LEVEL_ORDER]
        self.fig_epirank_map.legend(
            handles=legend_patches, loc='upper center',
            ncol=4, fontsize=9, frameon=False,
            title='EpiRank level', title_fontsize=10,
            bbox_to_anchor=(0.5, 0.99))

        self.fig_epirank_map.subplots_adjust(left=0.05, right=0.98,
                                              bottom=0.02, top=0.92)
        self.canvas_epirank_map.draw()

    # ---- Tab 11: EpiRank vs 疾病地图 ----
    def _draw_overlay_map(self):
        self.fig_overlay_map.clear()

        g = self.results['g']
        npos = self.results['npos']
        nodes_list = list(g.nodes())

        er_matrix = self.results['epidemic_risk']
        er_vals = np.array(er_matrix).flatten()
        er_rank = {nodes_list[i]: er_vals[i] for i in range(len(nodes_list))}
        er_arr = np.array(list(er_rank.values()))
        er_keys = list(er_rank.keys())
        er_breaks = head_tail_breaks(er_arr)
        er_labels = classify_by_breaks(er_arr, er_breaks) if len(er_breaks) >= 3 else ['NC'] * len(er_arr)
        er_label_map = {k: lab for k, lab in zip(er_keys, er_labels)}
        er_core_nodes = {k for k, lab in er_label_map.items() if lab != 'NC'}

        taipei_nodes = [s for s in nodes_list if g.nodes[s][KEY_DB_ID] in GTAIPEI_DB_IDS]
        if not taipei_nodes:
            self.canvas_overlay_map.draw()
            return

        tx = [npos[k][0] for k in taipei_nodes if k in npos]
        ty = [npos[k][1] for k in taipei_nodes if k in npos]
        if not tx or not ty:
            self.canvas_overlay_map.draw()
            return
        pad_x = (max(tx) - min(tx)) * 0.08 or 0.5
        pad_y = (max(ty) - min(ty)) * 0.08 or 0.5

        size_map = {'C-I': 120, 'C-II': 60, 'C-III': 30, 'NC': 15}
        legend_labels = {'NC': 'non-core', 'C-III': 'core-III',
                         'C-II': 'core-II', 'C-I': 'core-I'}

        disease_info = [
            ('(a) flu cases', self.results['flu_case_rank']),
            ('(b) EV cases', self.results['ev_case_rank']),
        ]

        from matplotlib.lines import Line2D

        for col, (title, case_dict) in enumerate(disease_info):
            ax = self.fig_overlay_map.add_subplot(1, 2, col + 1)

            all_vals = np.array(list(case_dict.values()))
            all_keys = list(case_dict.keys())
            breaks = head_tail_breaks(all_vals)
            if len(breaks) < 3:
                ax.set_title(title + '\n(insufficient data)')
                continue
            all_labels = classify_by_breaks(all_vals, breaks)
            global_label_map = {k: lab for k, lab in zip(all_keys, all_labels)}

            taipei_keys = [k for k in taipei_nodes if k in case_dict and k in npos]

            bg_x = [npos[k][0] for k in taipei_keys]
            bg_y = [npos[k][1] for k in taipei_keys]
            ax.scatter(bg_x, bg_y, s=60, c='#e0e0e0', edgecolors='#d0d0d0',
                       linewidths=0.3, zorder=1)

            for level in LEVEL_ORDER:
                xs = [npos[k][0] for k in taipei_keys if global_label_map.get(k) == level]
                ys = [npos[k][1] for k in taipei_keys if global_label_map.get(k) == level]
                if xs:
                    ax.scatter(xs, ys, s=size_map[level] * 1.5, c=LEVEL_COLORS[level],
                               edgecolors='gray', linewidths=0.3,
                               label=legend_labels[level],
                               zorder=2 + LEVEL_ORDER.index(level))

            core_in_taipei = [k for k in taipei_keys if k in er_core_nodes]
            if core_in_taipei:
                cx = [npos[k][0] for k in core_in_taipei]
                cy = [npos[k][1] for k in core_in_taipei]
                ax.scatter(cx, cy, s=180, facecolors='none', edgecolors='black',
                           linewidths=2, zorder=10, label='EpiRank core township')

            ax.set_xlim(min(tx) - pad_x, max(tx) + pad_x)
            ax.set_ylim(min(ty) - pad_y, max(ty) + pad_y)
            ax.set_title(title, fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel('Taipei Metropolitan Area', fontsize=10, labelpad=20)
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)

        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=LEVEL_COLORS[lv],
                   markersize=8, markeredgecolor='gray', label=legend_labels[lv])
            for lv in LEVEL_ORDER
        ]
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markersize=10, markeredgecolor='black', markeredgewidth=2,
                   label='EpiRank core township'))
        self.fig_overlay_map.legend(
            handles=legend_handles, loc='lower center',
            ncol=5, fontsize=8, frameon=False,
            title='disease severity levels', title_fontsize=9,
            bbox_to_anchor=(0.5, 0.02))

        self.fig_overlay_map.tight_layout(rect=[0, 0.08, 1, 1])
        self.canvas_overlay_map.draw()

    # ---- Tab 3: 相关性表 ----
    def _populate_evaluation_summary(self):
        """填充 Evaluation Summary 标签页，显示所有指标与疾病的评估指标汇总。"""
        if self.results is None or 'evaluation_summary' not in self.results:
            return

        summary = self.results['evaluation_summary']
        headers = ['Index', 'Disease', 'Pearson_r', 'Spearman_rho', 'Kendall_tau',
                   'Recall', 'Precision', 'F1']
        self.eval_table.setRowCount(len(summary))
        self.eval_table.setColumnCount(len(headers))
        self.eval_table.setHorizontalHeaderLabels(headers)

        for row, item in enumerate(summary):
            for col, key in enumerate(headers):
                val = item.get(key, '')
                if isinstance(val, float):
                    if np.isnan(val):
                        val = 'N/A'
                    else:
                        val = f"{val:.4f}"
                cell = QTableWidgetItem(str(val))
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.eval_table.setItem(row, col, cell)

        self.eval_table.resizeColumnsToContents()

    # ---- Tab 14: Transient Dynamics（M1）----
    def _draw_transient(self):
        """M1 暂态动力学：轨迹、到达时间、速度、SARS leading indicator。"""
        self.fig_transient.clear()

        r = self.results
        if r is None or r.get('transient_metrics') is None:
            ax = self.fig_transient.add_subplot(111)
            ax.text(0.5, 0.5, "No transient data. Re-run with record_trajectory=True.",
                    ha='center', va='center', fontsize=12, color='#888')
            ax.axis('off')
            self.canvas_transient.draw()
            return

        tm = r['transient_metrics']
        traj = tm['trajectory']
        arrival = tm['arrival_time']
        velocity = tm['velocity']
        er_star = tm['er_star']
        g = r['g']
        npos = r['npos']
        town_data = r['town_data']
        nodes_list = list(g.nodes())
        N = len(nodes_list)
        T = traj.shape[0]

        # 排序：按稳态 ER 降序，取前 50 个核心县做轨迹图
        top_k = 50
        order = np.argsort(-er_star)[:top_k]

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=self.fig_transient,
                      hspace=0.32, wspace=0.28,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

        # ── (a) ER_t 轨迹（前 50 核心县）──
        ax_a = self.fig_transient.add_subplot(gs[0, 0])
        cmap = matplotlib.colormaps.get_cmap('YlOrRd')
        for rank, idx in enumerate(order):
            ax_a.plot(np.arange(T), traj[:, idx],
                      color=cmap(1.0 - rank / top_k),
                      linewidth=0.8, alpha=0.75)
        ax_a.set_xlabel('iteration (day)', fontsize=9)
        ax_a.set_ylabel('ER_t', fontsize=9)
        ax_a.set_title(f'(a) ER_t trajectory — top {top_k} core townships '
                       f'(T={T})', fontsize=10)
        ax_a.grid(True, alpha=0.25)
        ax_a.tick_params(labelsize=7)

        # ── (b) 到达时间地图 ──
        ax_b = self.fig_transient.add_subplot(gs[0, 1])
        finite_mask = np.isfinite(arrival)
        vals_plot = arrival.copy()
        vals_plot[~finite_mask] = np.nan
        valid = vals_plot[~np.isnan(vals_plot)]
        if len(valid) > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            all_x = [npos[k][0] for k in nodes_list if k in npos]
            all_y = [npos[k][1] for k in nodes_list if k in npos]
            ax_b.scatter(all_x, all_y, s=6, c='#d8d8d8',
                         edgecolors='none', zorder=1)

            xs = [npos[nodes_list[i]][0] for i in range(N)
                  if nodes_list[i] in npos and np.isfinite(arrival[i])]
            ys = [npos[nodes_list[i]][1] for i in range(N)
                  if nodes_list[i] in npos and np.isfinite(arrival[i])]
            cs = [arrival[i] for i in range(N) if np.isfinite(arrival[i])]
            sc = ax_b.scatter(xs, ys, c=cs, s=22,
                              cmap='RdYlBu', vmin=vmin, vmax=vmax,
                              edgecolors='gray', linewidths=0.3,
                              zorder=3)
            cbar = self.fig_transient.colorbar(sc, ax=ax_b,
                                                fraction=0.035, pad=0.02)
            cbar.set_label('arrival iteration', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            n_inf = int((~finite_mask).sum())
            ax_b.set_title(f'(b) arrival time t_i (θ=0.5)  '
                           f'[{n_inf} N/A]', fontsize=10)
        else:
            ax_b.set_title('(b) arrival time — no finite values', fontsize=10)
        ax_b.set_aspect('equal')
        ax_b.axis('off')

        # ── (c) t_i vs SARS 病例数（leading indicator 验证）──
        ax_c = self.fig_transient.add_subplot(gs[1, 0])
        sars_rank = r['sars_case_rank']
        xs_c, ys_c = [], []
        for i, seq in enumerate(nodes_list):
            if np.isfinite(arrival[i]) and seq in sars_rank:
                xs_c.append(arrival[i])
                ys_c.append(sars_rank[seq])
        if len(xs_c) > 5:
            ax_c.scatter(xs_c, ys_c, s=18, c='steelblue',
                         alpha=0.65, edgecolors='gray', linewidths=0.3)
            pr, pp = st.pearsonr(xs_c, ys_c)
            sr, sp = st.spearmanr(xs_c, ys_c)
            ax_c.text(0.05, 0.95,
                      f"Pearson r = {pr:.3f} (p={pp:.3g})\n"
                      f"Spearman ρ = {sr:.3f} (p={sp:.3g})\n"
                      f"n = {len(xs_c)}",
                      transform=ax_c.transAxes, fontsize=8.5,
                      va='top', ha='left',
                      bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='#fffbe6',
                                edgecolor='#cccc88'))
            ax_c.set_xlabel('arrival iteration t_i', fontsize=9)
            ax_c.set_ylabel('SARS total cases', fontsize=9)
            ax_c.set_title('(c) t_i vs SARS cases (leading indicator check)',
                           fontsize=10)
            ax_c.grid(True, alpha=0.25)
        else:
            ax_c.text(0.5, 0.5, 'insufficient data',
                      ha='center', va='center', fontsize=11, color='#888')
            ax_c.set_title('(c) t_i vs SARS cases', fontsize=10)
        ax_c.tick_params(labelsize=7)

        # ── (d) 速度直方图 + 谱间隙文本框 ──
        ax_d = self.fig_transient.add_subplot(gs[1, 1])
        pos_vel = velocity[velocity > 0]
        if len(pos_vel) > 0:
            log_v = np.log10(pos_vel)
            ax_d.hist(log_v, bins=30, color='#d2691e',
                      edgecolor='white', linewidth=0.3, alpha=0.85)
            ax_d.set_xlabel(r'$\log_{10}(v_i)$  (risk velocity)',
                            fontsize=9)
            ax_d.set_ylabel('number of townships', fontsize=9)
        else:
            ax_d.text(0.5, 0.5, 'no positive velocity',
                      ha='center', va='center', fontsize=11, color='#888')
            ax_d.set_xlabel('risk velocity', fontsize=9)
        ax_d.set_title('(d) risk velocity distribution', fontsize=10)
        ax_d.tick_params(labelsize=7)

        # 叠加谱间隙/Kemeny 文本框
        info_text = (
            f"{'abs gap |λ₂|/|λ₁|':<20} = {tm['spectral_gap']:.5f}\n"
            f"{'Kemeny Σ 1/(1-λᵣ)':<20} = {tm['kemeny_constant']:.2f}\n"
            f"{'|λ₁|':<20} = {tm['abs_lambda1']:.6f}\n"
            f"{'|λ₂|':<20} = {tm['abs_lambda2']:.6f}"
        )
        ax_d.text(0.98, 0.97, info_text,
                  transform=ax_d.transAxes,
                  fontsize=8.5, va='top', ha='right',
                  family='monospace',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='#eef6ff',
                            edgecolor='#88aacc'))

        self.canvas_transient.draw()

    # ---- Tab 15: Attribution（M2）----
    def _draw_attribution(self):
        """M2 方向性风险归因：净流入/流出、边流量矩阵、top-k 源、闭式校验。"""
        self.fig_attribution.clear()

        r = self.results
        ad = r.get('attribution_data') if r else None
        if ad is None:
            ax = self.fig_attribution.add_subplot(111)
            ax.text(0.5, 0.5, "No attribution data. Re-run with record_attribution=True.",
                    ha='center', va='center', fontsize=12, color='#888')
            ax.axis('off')
            self.canvas_attribution.draw()
            return

        g = r['g']
        npos = r['npos']
        town_data = r['town_data']
        nodes_list = list(g.nodes())
        N = len(nodes_list)

        F_net = ad['F_net']
        net_in = ad['net_inflow']
        net_out = ad['net_outflow']
        top_sources = ad['top_sources']
        residual = ad['closed_form_residual']

        # 县名（用于标注）
        def label_of(i):
            seq = nodes_list[i]
            db = g.nodes[seq][KEY_DB_ID]
            return f"{town_data[db][KEY_COUNTY]}{town_data[db][KEY_TOWN]}"

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=self.fig_attribution,
                      hspace=0.34, wspace=0.30,
                      left=0.07, right=0.97, top=0.93, bottom=0.08)

        # ── (a) 净流入 vs 净流出 散点 ──
        ax_a = self.fig_attribution.add_subplot(gs[0, 0])
        ax_a.scatter(net_out, net_in, s=18, c='steelblue',
                     alpha=0.6, edgecolors='gray', linewidths=0.3)
        # 对角线
        lim = max(net_in.max(), net_out.max()) * 1.05
        if lim <= 0:
            lim = 1.0
        ax_a.plot([0, lim], [0, lim], 'gray', linestyle='--',
                  linewidth=0.8, alpha=0.7)
        # 标注 top-5 净流入 & top-5 净流出（自动防重叠）
        top_in_idx = np.argsort(-net_in)[:5]
        top_out_idx = np.argsort(-net_out)[:5]
        from adjustText import adjust_text
        texts = []
        for i in top_in_idx:
            texts.append(ax_a.text(net_out[i], net_in[i], label_of(i), fontsize=6.5, color='darkred'))
        for i in top_out_idx:
            if i not in top_in_idx:
                texts.append(ax_a.text(net_out[i], net_in[i], label_of(i), fontsize=6.5, color='darkblue'))
        adjust_text(texts, ax=ax_a, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        ax_a.set_xlabel('net outflow (Σ_j F[i,j])', fontsize=9)
        ax_a.set_ylabel('net inflow (Σ_j F[j,i])', fontsize=9)
        ax_a.set_title('(a) net inflow vs net outflow', fontsize=10)
        ax_a.grid(True, alpha=0.25)
        ax_a.tick_params(labelsize=7)

        # ── (b) F_net 矩阵热图（log 尺度，取非零部分）──
        ax_b = self.fig_attribution.add_subplot(gs[0, 1])
        F_plot = F_net.copy()
        np.fill_diagonal(F_plot, 0)
        im = ax_b.imshow(F_plot, aspect='auto', cmap='YlOrRd',
                         interpolation='nearest',
                         vmin=0, vmax=np.percentile(F_plot, 99))
        cbar = self.fig_attribution.colorbar(im, ax=ax_b,
                                              fraction=0.04, pad=0.02)
        cbar.set_label('cumulative flow F[i,j]', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax_b.set_xlabel('source j', fontsize=9)
        ax_b.set_ylabel('target i', fontsize=9)
        ax_b.set_title('(b) F_net[i,j] heatmap (self-loop removed)', fontsize=10)
        ax_b.tick_params(labelsize=6)

        # ── (c) Top-5 核心县 的 top-5 源 条形图 ──
        ax_c = self.fig_attribution.add_subplot(gs[1, 0])
        er_star = r['epidemic_risk'].flatten()
        top5 = np.argsort(-er_star)[:5]

        y_positions = []
        labels_y = []
        bar_colors = []
        bar_vals = []
        y = 0
        for i in top5:
            srcs = top_sources.get(i, [])
            if not srcs:
                continue
            # 标题行
            labels_y.append(f"[{label_of(i)}]")
            y_positions.append(y)
            bar_vals.append(0)
            bar_colors.append('#ffffff')
            y += 1
            for (j, flow) in srcs:
                labels_y.append(f"  {label_of(j)} →")
                y_positions.append(y)
                bar_vals.append(flow)
                bar_colors.append('#5a9bd4')
                y += 1
            y += 0.3

        if bar_vals:
            ax_c.barh(y_positions, bar_vals, color=bar_colors,
                      edgecolor='white', linewidth=0.3)
            ax_c.set_yticks(y_positions)
            ax_c.set_yticklabels(labels_y, fontsize=5)
            ax_c.invert_yaxis()
        ax_c.set_xlabel('cumulative flow to target', fontsize=9)
        ax_c.set_title('(c) top-5 sources for top-5 core townships',
                       fontsize=10)
        ax_c.tick_params(axis='x', labelsize=7)
        ax_c.grid(True, alpha=0.2, axis='x')

        # ── (d) 净流入地图 + 闭式校验 ──
        ax_d = self.fig_attribution.add_subplot(gs[1, 1])
        # 背景节点（浅灰）
        all_x = [npos[k][0] for k in nodes_list if k in npos]
        all_y = [npos[k][1] for k in nodes_list if k in npos]
        ax_d.scatter(all_x, all_y, s=4, c='#e0e0e0',
                     edgecolors='none', zorder=1)
        # 叠加净流入
        xs = [npos[nodes_list[i]][0] for i in range(N) if nodes_list[i] in npos]
        ys = [npos[nodes_list[i]][1] for i in range(N) if nodes_list[i] in npos]
        cs = [net_in[i] for i in range(N) if nodes_list[i] in npos]
        if cs:
            vmin, vmax = 0.0, float(np.percentile(cs, 99)) or 1.0
            sc = ax_d.scatter(xs, ys, c=cs, s=25, cmap='YlOrRd',
                              vmin=vmin, vmax=vmax,
                              edgecolors='gray', linewidths=0.3,
                              zorder=3)
            cbar2 = self.fig_attribution.colorbar(sc, ax=ax_d,
                                                    fraction=0.035, pad=0.02)
            cbar2.set_label('net inflow', fontsize=8)
            cbar2.ax.tick_params(labelsize=7)
        ax_d.set_aspect('equal')
        ax_d.axis('off')
        ax_d.set_title('(d) net inflow map', fontsize=10)

        # 闭式校验文本框
        info_text = f"closed-form residual = {residual:.3e}"
        ax_d.text(0.02, 0.98, info_text,
                  transform=ax_d.transAxes,
                  fontsize=8.5, va='top', ha='left',
                  family='monospace',
                  bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='#eef6ff',
                            edgecolor='#88aacc'))

        self.canvas_attribution.draw()

    # ---- 自动储存 ----
    def _auto_save_results(self):
        r = self.results
        d = r['d']
        iters = r['iterations']
        base_name = f"ERA_result_reverse_seqday_d_{d}_loops_{iters}"

        try:
            xlsx_path = os.path.join(self.data_dir, base_name + '.xlsx')
            self._save_excel_to_path(xlsx_path)
            self._log(f"Auto-saved: {xlsx_path}")

            json_path = os.path.join(self.data_dir, base_name + '.json')
            self._save_evaluation_summary_json(json_path)
            self._log(f"Auto-saved: {json_path}")

            self._log(f"\n--- All 2 output files saved to: {self.data_dir} ---")
        except Exception as e:
            self._log(f"\nWARNING: Auto-save error: {e}")

    def _save_excel_to_path(self, path):
        r = self.results
        wb = Workbook()
        ws = wb.active
        ws.title = "EpiRank Results"

        g = r['g']
        town_data = r['town_data']
        ER_avg = r['ER_avg']
        ER_std = r['ER_std']
        ER_tot = r['ER_tot']

        info_font = Font(name='Arial Narrow', color='8B0000', bold=True)
        title_font = Font(name='Arial Narrow', color='8B0000', bold=True)
        body_font = Font(name='Arial Narrow', color='00008B')

        in_degrees = dict(g.in_degree())
        out_degrees = dict(g.out_degree())
        in_vals = list(in_degrees.values())
        out_vals = list(out_degrees.values())

        info_rows = [
            f"number of nodes = {g.number_of_nodes()}",
            f"number of edges = {g.number_of_edges()}",
            f"average in-degree = {np.mean(in_vals)}",
            f"STD of in-degree = {np.std(in_vals)}",
            f"average out-degree = {np.mean(out_vals)}",
            f"STD of out-degree = {np.std(out_vals)}",
            "", "", "", "",
            f"number of selfloop edges = {nx.number_of_selfloops(g)}",
            f"number of epidemic analysis loops = {r['iterations']}",
            f"parameter.d = {round(r['d'], 2)}",
            f"mode = Reverse Sequential-day (evening -> morning, exFac at day end)",
        ]
        for i, text in enumerate(info_rows):
            ws.cell(row=i + 1, column=1, value=text).font = info_font

        headers = ['seq no', 'post code', 'db ID', 'county', 'town',
                    'pos.x', 'pos.y', 'population', 'area (km^2)',
                    'density (D)', 'normalized D',
                    'age  0 ~ 14', 'age 15 ~ 64', 'age 65+',
                    'ERV', 'ERP (%)',
                    'C.local', 'C.out', 'C.in', 'C.out-towns', 'C.in-towns',
                    'R.zone', 'page rank', 'hits.hub', 'hits.authority',
                    'flu cases', 'ev cases', 'sars cases']
        for j, h in enumerate(headers):
            cell = ws.cell(row=1, column=j + 2, value=h)
            cell.font = title_font
            cell.alignment = Alignment(horizontal='center')

        ws.cell(row=1, column=30, value='AVG+2STD =').font = info_font
        ws.cell(row=1, column=31, value=round(ER_avg + 2 * ER_std, 5)).font = info_font
        ws.cell(row=1, column=33, value='AVG+STD =').font = info_font
        ws.cell(row=1, column=34, value=round(ER_avg + ER_std, 5)).font = info_font
        ws.cell(row=1, column=36, value='AVG=').font = info_font
        ws.cell(row=1, column=37, value=round(ER_avg, 5)).font = info_font
        ws.cell(row=1, column=39, value='AVG-STD =').font = info_font
        ws.cell(row=1, column=40, value=round(ER_avg - ER_std, 5)).font = info_font
        ws.cell(row=1, column=42, value='AVG-2STD =').font = info_font
        ws.cell(row=1, column=43, value=round(ER_avg - 2 * ER_std, 5)).font = info_font

        row_idx = 2
        risk_keys = ['vh', 'h', 'm', 'l', 'vl']
        risk_cols = {'vh': 30, 'h': 33, 'l': 39, 'm': 36, 'vl': 42}
        risk_row_idx = {k: 2 for k in risk_keys}

        for item in r['table_data']:
            seq_no = item['seq_no']
            db_ID = item['db_ID']
            td = town_data[db_ID]
            erv = item['ERV']

            vals = [
                seq_no, item['post_code'], db_ID,
                td[KEY_COUNTY], td[KEY_TOWN],
                td[KEY_POS_XY][0], td[KEY_POS_XY][1],
                td[KEY_POPULATION],
                round(td[KEY_AREA], 2), round(td[KEY_DENSITY], 2),
                round(td[KEY_NORMALIZED_DENSITY], 6),
                str(round(td[KEY_AGE_0_14], 2)) + '%',
                str(round(td[KEY_AGE_15_64], 2)) + '%',
                str(round(td[KEY_AGE_65], 2)) + '%',
                round(erv, 5),
                str(round(100.0 * erv / ER_tot, 2)) + '%' if ER_tot > 0 else '0%',
                td[KEY_LOCAL_COMMUTER_TYPE1],
                td[KEY_OUT_COMMUTER_TYPE1], td[KEY_IN_COMMUTER_TYPE1],
                g.in_degree(seq_no), g.out_degree(seq_no),
                td.get('railroad_zone', 0),
                round(r['page_rank'].get(seq_no, 0), 6),
                round(r['hub_rank'].get(seq_no, 0), 6),
                round(r['authority_rank'].get(seq_no, 0), 6),
                td.get(KEY_FLU_TOTAL_CASES, 0),
                td.get(KEY_EV_AVERAGE_CASES, 0),
                td.get(KEY_SARS_TOTAL_CASES, 0),
            ]
            for j, v in enumerate(vals):
                ws.cell(row=row_idx, column=j + 2, value=v).font = body_font

            if erv >= ER_avg + 2 * ER_std:
                key = 'vh'
            elif erv >= ER_avg + ER_std:
                key = 'h'
            elif erv >= ER_avg:
                key = 'm'
            elif erv >= ER_avg - ER_std:
                key = 'l'
            else:
                key = 'vl'
            c = risk_cols[key]
            ri = risk_row_idx[key]
            ws.cell(row=ri, column=c, value=td[KEY_COUNTY]).font = body_font
            ws.cell(row=ri, column=c + 1, value=td[KEY_TOWN]).font = body_font
            risk_row_idx[key] += 1

            row_idx += 1

        sr = row_idx
        ER_rank = r['ER_rank']
        pop_rank = r['pop_rank']
        flu_case_rank = r['flu_case_rank']
        ev_case_rank = r['ev_case_rank']
        sars_case_rank = r['sars_case_rank']
        page_rank = r['page_rank']
        hub_rank = r['hub_rank']
        authority_rank = r['authority_rank']

        nodes_list = list(g.nodes())
        gTaipei_ER_rank = {s: ER_rank[s] for s in nodes_list if g.nodes[s][KEY_DB_ID] in GTAIPEI_DB_IDS}
        gTaipei_sars = {s: sars_case_rank[s] for s in nodes_list if g.nodes[s][KEY_DB_ID] in GTAIPEI_DB_IDS}
        gTaipei_pr_ids = set(range(0, 29)) | set(range(303, 310)) | set(range(330, 342))
        gTaipei_page = {k: v for k, v in page_rank.items() if k in gTaipei_pr_ids}
        gTaipei_hub = {k: v for k, v in hub_rank.items() if k in gTaipei_pr_ids}
        gTaipei_auth = {k: v for k, v in authority_rank.items() if k in gTaipei_pr_ids}

        # ── Population vs 网络指数 correlations (cols 8-9) ──
        pop_corr_data = [
            ('Population vs EpiRank:  Pearson',  get_pearson_cor(pop_rank, ER_rank)[0]),
            ('Population vs EpiRank:  Spearman', get_spearman_cor(pop_rank, ER_rank)[0]),
            ('Population vs Flu:      Pearson',  get_pearson_cor(pop_rank, flu_case_rank)[0]),
            ('Population vs Flu:      Spearman', get_spearman_cor(pop_rank, flu_case_rank)[0]),
            ('Population vs SARS:     Pearson',  get_pearson_cor(pop_rank, sars_case_rank)[0]),
            ('Population vs SARS:     Spearman', get_spearman_cor(pop_rank, sars_case_rank)[0]),
        ]
        for i, (lbl, val) in enumerate(pop_corr_data):
            ws.cell(row=sr + i, column=8, value=lbl).font = body_font
            ws.cell(row=sr + i, column=9, value=val).font = body_font

        # ── ER stats (cols 15-16) ──
        ws.cell(row=sr, column=15, value='EpiRank total = ').font = body_font
        ws.cell(row=sr, column=16, value=round(ER_tot, 5)).font = body_font
        ws.cell(row=sr + 1, column=15, value='EpiRank mean  = ').font = body_font
        ws.cell(row=sr + 1, column=16, value=round(ER_avg, 5)).font = body_font
        ws.cell(row=sr + 2, column=15, value='EpiRank SD    = ').font = body_font
        ws.cell(row=sr + 2, column=16, value=round(ER_std, 5)).font = body_font

        labels = [
            'pearson  (*,  EV)', 'spearman (*,  EV)',
            'pearson  (*, Flu)', 'spearman (*, Flu)',
            'pearson  (*, SARS)', 'spearman (*, SARS)', 'kendall. (*, SARS)',
        ]
        for i, lbl in enumerate(labels):
            ws.cell(row=sr + i, column=23, value=lbl).font = body_font

        ws.cell(row=sr, column=24, value=get_pearson_cor(page_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 1, column=24, value=get_spearman_cor(page_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 2, column=24, value=get_pearson_cor(page_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 3, column=24, value=get_spearman_cor(page_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 4, column=24, value=get_pearson_cor(gTaipei_page, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 5, column=24, value=get_spearman_cor(gTaipei_page, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 6, column=24, value=get_kendalltau_cor(gTaipei_page, gTaipei_sars)[0]).font = body_font

        ws.cell(row=sr, column=25, value=get_pearson_cor(hub_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 1, column=25, value=get_spearman_cor(hub_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 2, column=25, value=get_pearson_cor(hub_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 3, column=25, value=get_spearman_cor(hub_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 4, column=25, value=get_pearson_cor(gTaipei_hub, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 5, column=25, value=get_spearman_cor(gTaipei_hub, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 6, column=25, value=get_kendalltau_cor(gTaipei_hub, gTaipei_sars)[0]).font = body_font

        ws.cell(row=sr, column=26, value=get_pearson_cor(authority_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 1, column=26, value=get_spearman_cor(authority_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 2, column=26, value=get_pearson_cor(authority_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 3, column=26, value=get_spearman_cor(authority_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 4, column=26, value=get_pearson_cor(gTaipei_auth, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 5, column=26, value=get_spearman_cor(gTaipei_auth, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 6, column=26, value=get_kendalltau_cor(gTaipei_auth, gTaipei_sars)[0]).font = body_font

        ws.cell(row=sr, column=28, value=get_pearson_cor(ER_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 1, column=28, value=get_spearman_cor(ER_rank, ev_case_rank)[0]).font = body_font
        ws.cell(row=sr + 2, column=27, value=get_pearson_cor(ER_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 3, column=27, value=get_spearman_cor(ER_rank, flu_case_rank)[0]).font = body_font
        ws.cell(row=sr + 4, column=29, value=get_pearson_cor(gTaipei_ER_rank, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 5, column=29, value=get_spearman_cor(gTaipei_ER_rank, gTaipei_sars)[0]).font = body_font
        ws.cell(row=sr + 6, column=29, value=get_kendalltau_cor(gTaipei_ER_rank, gTaipei_sars)[0]).font = body_font

        for c, lbl in [(24, 'page rank'), (25, 'hits.hub'), (26, 'hits.authority'),
                         (27, 'flu cases'), (28, 'ev cases'), (29, 'sars cases')]:
            ws.cell(row=sr + 7, column=c, value=lbl).font = title_font

        # ── 新增 Evaluation Summary 工作表 ──
        ws2 = wb.create_sheet("Evaluation Summary")
        headers_eval = ['Index', 'Disease', 'Pearson_r', 'Spearman_rho', 'Kendall_tau',
                        'Recall', 'Precision', 'F1']
        for col, h in enumerate(headers_eval, 1):
            cell = ws2.cell(row=1, column=col, value=h)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')

        for row_idx, item in enumerate(r['evaluation_summary'], start=2):
            ws2.cell(row=row_idx, column=1, value=item['Index'])
            ws2.cell(row=row_idx, column=2, value=item['Disease'])
            ws2.cell(row=row_idx, column=3, value=item['Pearson_r'])
            ws2.cell(row=row_idx, column=4, value=item['Spearman_rho'])
            ws2.cell(row=row_idx, column=5, value=item['Kendall_tau'])
            ws2.cell(row=row_idx, column=6, value=item['Recall'])
            ws2.cell(row=row_idx, column=7, value=item['Precision'])
            ws2.cell(row=row_idx, column=8, value=item['F1'])

        # 简单调整列宽
        for col in range(1, 9):
            ws2.column_dimensions[get_column_letter(col)].width = 15

        wb.save(path)

    def _compute_core_labels_dict(self):
        """为空间重叠分析导出每个指标（含疾病）的乡镇级核心等级标签。"""
        r = self.results
        g = r['g']
        nodes = list(g.nodes())
        town_data = r['town_data']

        indices = {
            'EpiRank':        r['ER_rank'],
            'PageRank':       r['page_rank'],
            'HITS-Hub':       r['hub_rank'],
            'HITS-Authority': r['authority_rank'],
            'Population':     r['pop_rank'],
            'Flu':            r['flu_case_rank'],
            'EV':             r['ev_case_rank'],
            'SARS':           r['sars_case_rank'],
        }

        result = {}
        for name, dict_ in indices.items():
            values = np.array([dict_.get(n, 0) for n in nodes], dtype=float)
            breaks = head_tail_breaks(values, 3)
            labels = (classify_by_breaks(values, breaks)
                      if len(breaks) > 0 else ['NC'] * len(values))

            per_db = {}
            for i, seq in enumerate(nodes):
                db = g.nodes[seq][KEY_DB_ID]
                td = town_data[db]
                per_db[str(db)] = {
                    'level':  labels[i],
                    'score':  round(float(values[i]), 10),
                    'county': td[KEY_COUNTY],
                    'town':   td[KEY_TOWN],
                }
            result[name] = per_db
        return result

    def _save_evaluation_summary_json(self, path):
        r = self.results
        eval_fields = ['Index', 'Disease', 'Pearson_r', 'Spearman_rho',
                    'Kendall_tau', 'Recall', 'Precision', 'F1']

        def round_val(v):
            # NaN (样本不足/常量数组时 pearsonr 等返回的 N/A) → None → JSON null
            if isinstance(v, float) and np.isnan(v):
                return None
            return round(v, 4) if isinstance(v, (int, float)) else v

        eval_data = []
        for item in r.get('evaluation_summary', []):
            row = {}
            for k in eval_fields:
                row[k] = round_val(item.get(k))   # 顺便用 get() 更稳健
            eval_data.append(row)

        payload = {
            'metadata': {
                # 注意：两份文件的 mode 值不同，保持原样即可
                'mode':       'reverse_sequential_day',   # 或 'sequential_day'
                'd':          r['d'],
                'iterations': r['iterations'],
            },
            'evaluation_summary': eval_data,
            'core_labels':        self._compute_core_labels_dict(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 敏感度分析（扫描 d）----
    def _run_sensitivity(self):
        if not self.results:
            QMessageBox.information(self, "No Data", "Please run EpiRank first.")
            return

        self.btn_run.setEnabled(False)
        self.btn_sensitivity.setEnabled(False)
        self.progress_bar.setValue(0)
        self._log("\n=== Starting Sensitivity Analysis (by d) ===")
        self._log("Computing EpiRank for 20 d values...")

        self.sensitivity_worker = SensitivityWorker(
            self.data_dir,
            self.spin_loops.value()
        )
        self.sensitivity_worker.progress.connect(self._on_sensitivity_progress)
        self.sensitivity_worker.log_message.connect(self._log)
        self.sensitivity_worker.finished_ok.connect(self._on_sensitivity_ok)
        self.sensitivity_worker.finished_err.connect(self._on_sensitivity_err)
        self.sensitivity_worker.start()
        self._update_status("Running sensitivity analysis...")

    def _on_sensitivity_progress(self, cur, total):
        if total > 0:
            self.progress_bar.setValue(int(100 * cur / total))

    def _on_sensitivity_ok(self, results):
        self.btn_run.setEnabled(True)
        self.btn_sensitivity.setEnabled(True)
        self.progress_bar.setValue(100)
        self.sensitivity_results = results
        self._log("Sensitivity analysis completed successfully.")
        self._update_status("Sensitivity analysis completed.")
        self._draw_sensitivity()
        self.tabs.setCurrentIndex(13)

    def _on_sensitivity_err(self, err_msg):
        self.btn_run.setEnabled(True)
        self.btn_sensitivity.setEnabled(True)
        self.progress_bar.setValue(0)
        self._log(f"\nSensitivity Analysis ERROR:\n{err_msg}")
        self._update_status("Sensitivity analysis error.")
        QMessageBox.critical(self, "Sensitivity Analysis Error", str(err_msg)[:500])

    def _draw_sensitivity(self):
        self.fig_sensitivity.clear()

        sr = self.sensitivity_results
        d_vals = sr['d_values']

        matrices = [
            ("(a) flu (Pearson's R)", sr['flu_pearson']),
            ("(b) flu (Spearman's Rho)", sr['flu_spearman']),
            ("(c) EV (Pearson's R)", sr['ev_pearson']),
            ("(d) EV (Spearman's Rho)", sr['ev_spearman']),
        ]

        for idx, (title, values) in enumerate(matrices):
            ax = self.fig_sensitivity.add_subplot(2, 2, idx + 1)
            ax.plot(d_vals, values, 'o-', color='steelblue', linewidth=1.5, markersize=4)
            ax.set_xlabel('damping factor (d)', fontsize=9)
            ax.set_ylabel('correlation', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        self.fig_sensitivity.tight_layout()
        self.canvas_sensitivity.draw()

    # ---- 汇出 Excel ----
    def _export_excel(self):
        if not self.results:
            QMessageBox.information(self, "No Data", "Please run EpiRank first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results",
            f"ERA_result_reverse_seqday_d_{self.results['d']}.xlsx",
            "Excel Files (*.xlsx)")
        if not path:
            return

        try:
            self._save_excel_to_path(path)
            self._log(f"\nResults exported to: {path}")
            self._update_status(f"Exported to {path}")
            QMessageBox.information(self, "Export Complete", f"Results saved to:\n{path}")
        except Exception as e:
            self._log(f"\nExport error: {e}")
            QMessageBox.critical(self, "Export Error", str(e)[:500])

    # ---- 存储圖表 ----
    def _save_figure(self):
        tab_idx = self.tabs.currentIndex()
        fig_map = {
            1: (self.fig_network, "network_map"),
            4: (self.fig_analysis, "commuter_flow"),
            5: (self.fig_disease, "frequency_distributions"),
            6: (self.fig_epirank_dist, "frequency_distribution"),
            7: (self.fig_epirank_vs, "epirank_vs_disease"),
            8: (self.fig_index_comp, "index_comparison"),
            9: (self.fig_disease_map, "disease_map"),
            10: (self.fig_epirank_map, "epirank_map"),
            11: (self.fig_overlay_map, "epirank_vs_disease_map"),
            13: (self.fig_sensitivity, "sensitivity_analysis"),
            14: (self.fig_transient, "transient_dynamics"),
            15: (self.fig_attribution, "attribution"),
        }
        if tab_idx not in fig_map:
            QMessageBox.information(self, "Save Chart", "Please switch to a chart tab first.")
            return

        fig, default_name = fig_map[tab_idx]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", f"{default_name}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight')
                self._log(f"Chart saved to: {path}")
                self._update_status(f"Chart saved to {path}")
            except Exception as e:
                self._log(f"Error saving chart: {e}")
                QMessageBox.critical(self, "Save Error", str(e)[:500])


# ============================================================
# 入口
# ============================================================

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)
    window = EpiRankMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()