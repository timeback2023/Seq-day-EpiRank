import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats as st

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QScrollArea,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QBrush


# ==================== 数据文件配置（修改这里来切换比较对象） ====================
FOLDER_A = r'D:\Python\EpiRank\EpiRank_原'
PREFIX_A = 'ERA_result_d_'
FOLDER_B = r'D:\Python\EpiRank\EpiRank_改'
PREFIX_B = 'ERA_result_seqday_d_'


def _derive_name(prefix):
    m = re.search(r'ERA_result_(.+?)_d_', prefix)
    if m:
        name = m.group(1)
        return name.replace('_', ' ').title() if name else 'Synday'
    if prefix.startswith('ERA_result_') and prefix.endswith('_d_'):
        return 'Synday'
    return 'Unknown'


MODEL_A_NAME = _derive_name(PREFIX_A)
MODEL_B_NAME = _derive_name(PREFIX_B)
# =============================================================================

METRIC_KEYS = ['Pearson_r', 'Spearman_rho', 'Kendall_tau',
               'Recall', 'Precision', 'F1']
METRIC_LABELS = {
    'Pearson_r':    'Pearson r',
    'Spearman_rho': 'Spearman ρ',
    'Kendall_tau':  'Kendall τ',
    'Recall':       'Recall',
    'Precision':    'Precision',
    'F1':           'F1',
}

# 空间重叠比较所采用的「核心」级别定义
OVERLAP_LEVEL_SETS = [
    ('C-I 核心',           {'C-I'}),
    ('C-II 核心',          {'C-II'}),
    ('C-III 核心',         {'C-III'}),
    ('全部核心 (I+II+III)', {'C-I', 'C-II', 'C-III'}),
    ('NC 非核心',          {'NC'}),
]


def find_latest(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    if not files:
        raise FileNotFoundError(f"{folder} 中找不到 {prefix}*.json")
    return max(files, key=os.path.getmtime)


def load_json(path):
    """兼容两种 JSON：
       - 新格式：{metadata, evaluation_summary, core_labels}
       - 旧格式：纯列表 [ {Index, Disease, ...}, ... ]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'evaluation_summary' in data:
        return {
            'summary':     data['evaluation_summary'],
            'core_labels': data.get('core_labels', {}),
            'metadata':    data.get('metadata', {}),
        }
    if isinstance(data, list):
        return {'summary': data, 'core_labels': {}, 'metadata': {}}
    raise ValueError(f"无法识别的 JSON 格式：{path}")


def build_lookup(summary_list):
    return {(entry['Index'], entry['Disease']): entry for entry in summary_list}


def get_val(lookup, key, metric):
    # 从 lookup 取 (Index, Disease) 对应 metric 的数值。
 
    entry = lookup.get(key)
    if not entry or metric not in entry:
        return None
    v = entry[metric]
    if v is None:                       # JSON null
        return None
    if isinstance(v, str) and v.strip().lower() in ('nan', 'none', ''):
        return None                     # 兼容字符串形式的缺失值
    return v


# ==================== 空间重叠计算 ====================

def compute_overlap_metrics(labels_a, labels_b, positive_levels):
    """计算两个模型在指定「核心」级别集合下的空间重合程度。

    labels_a / labels_b : { db_id_str: {'level': 'C-I'|'C-II'|'C-III'|'NC', ...} }
    positive_levels     : set，例如 {'C-I'} 或 {'C-I','C-II','C-III'}

    返回：核心区域集合大小、交集、并集、IoU、Dice、双向覆盖率。
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
    cov_a = n_i / n_a if n_a > 0 else 0.0   # A 的核心有多少被 B 也识别为核
    cov_b = n_i / n_b if n_b > 0 else 0.0   # B 的核心有多少被 A 也识别为核

    return {
        'n_a': n_a, 'n_b': n_b, 'n_i': n_i, 'n_u': n_u,
        'iou': iou, 'dice': dice,
        'cov_a': cov_a, 'cov_b': cov_b,
        'set_a': set_a, 'set_b': set_b, 'inter': inter,
    }


def _fmt(v):
    return f'{v:.4f}'


# ==================== Steiger 检验 ====================
# 比较同一批样本上、共享「疾病」这一个变量的两个相关系数
# （A 模型 vs B 模型）是否有显著差异。Steiger (1980) / Williams (1959)：
#
#   det R = 1 - r12² - r13² - r23² + 2·r12·r13·r23   （三变量相关矩阵的行列式）
#   t = (r12 - r13) · sqrt( (n-3)(1 + r23) / (2·det R) )
#   df = n - 3
#
# r12 = A 指标 vs 疾病，r13 = B 指标 vs 疾病，r23 = A 指标 vs B 指标。
# 三个相关都必须来自同一批样本、同一口径，否则公式不成立。
#
# det R <= 0 表示这三个两两相关无法同时成立（即不存在对应的相关矩阵），
# 此时公式无意义，按 N/A 处理。
#
# Spearman 那一列先把三条序列取秩再算 Pearson r（即 scipy 的定义），
# 再套同一公式——这是把 Steiger 公式套到 ρ 上的常见近似做法。

STEIGER_METRICS = ('Pearson_r', 'Spearman_rho')


def _pearson_r(x, y):
    """两条序列的 Pearson r；任一方为常量时返回 nan。"""
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def steiger_pvalue(r_12, r_13, r_23, n):
    """Steiger (1980) / Williams (1959) 双尾检验。

    用于比较共享同一个变量的两个相关系数 r_12 与 r_13
    （此处共享变量是疾病，A、B 是两个模型）。

        t = (r12 - r13) * sqrt( (n-3)(1+r23) / (2*det R) )
        det R = 1 - r12² - r13² - r23² + 2 r12 r13 r23
        df = n - 3
    """
    if n is None or n < 4:
        return None
    if any(v is None or not math.isfinite(v) for v in (r_12, r_13, r_23)):
        return None

    det_r = (1.0 - r_12 ** 2 - r_13 ** 2 - r_23 ** 2
             + 2.0 * r_12 * r_13 * r_23)
    if not math.isfinite(det_r) or det_r <= 0:
        # det R <= 0 意味着三个相关无法同时成立（不可能是同一个总体的
        # 相关矩阵），此时公式无意义，返回 None 由上层标记为 N/A。
        return None

    denom = 2.0 * det_r / (1.0 + r_23)
    if not math.isfinite(denom) or denom <= 0:
        return None

    t = (r_12 - r_13) * math.sqrt((n - 3) / denom)
    p = 2.0 * st.t.sf(abs(t), n - 3)
    if not math.isfinite(p):
        return None
    return min(max(p, 0.0), 1.0)


def steiger_p_for(labels_a, labels_b, index, disease, metric):
    """从两份 core_labels 取对齐的三条序列，算 Steiger 检验的 p 值。"""
    if metric not in STEIGER_METRICS:
        return None
    if not (labels_a and labels_b):
        return None
    src_a = labels_a.get(index)
    src_b = labels_b.get(index)
    dis = labels_a.get(disease) or labels_b.get(disease)
    if not (src_a and src_b and dis):
        return None

    ids = sorted(set(src_a) & set(src_b) & set(dis), key=lambda x: int(x))
    n = len(ids)
    if n < 4:
        return None
    try:
        va = np.array([src_a[i]['score'] for i in ids], dtype=float)
        vb = np.array([src_b[i]['score'] for i in ids], dtype=float)
        vd = np.array([dis[i]['score'] for i in ids], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(va).all() and np.isfinite(vb).all()
            and np.isfinite(vd).all()):
        return None

    if metric == 'Spearman_rho':
        va, vb, vd = st.rankdata(va), st.rankdata(vb), st.rankdata(vd)
    return steiger_pvalue(_pearson_r(va, vd), _pearson_r(vb, vd),
                          _pearson_r(va, vb), n)


def _fmt_p(p):
    if p is None:
        return 'N/A'
    if p < 1e-4:
        return '<0.0001'
    return f'{p:.4f}'


# ==================== Bootstrap 置信区间 ====================
# Recall / Precision / F1 不是相关系数，Steiger 检验不适用。
# 改为按乡镇有放回重采样，得到 Δ（= B - A）的经验分布与 95% 百分位区间。
# 一次重采样同时给出三项指标的分布，因此每个 (指标, 疾病) 只跑一次。

BOOTSTRAP_METRICS = ('Recall', 'Precision', 'F1')


def bootstrap_metric_diff(label_a, label_b, disease_label,
                           n_boot=2000, seed=0):
    """按乡镇重采样，得到 ΔRecall / ΔPrecision / ΔF1 的经验分布。

    返回 {metric: (mean, (lo, hi))}，区间为 2.5 / 97.5 百分位。
    """
    rng = np.random.default_rng(seed)
    ids = sorted(set(label_a) & set(label_b) & set(disease_label))
    n = len(ids)
    if n == 0:
        return None
    deltas = {'Recall': [], 'Precision': [], 'F1': []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [ids[i] for i in idx]
        a_pred = {k for k in sample if label_a[k]['level'] != 'NC'}
        b_pred = {k for k in sample if label_b[k]['level'] != 'NC'}
        actual = {k for k in sample if disease_label[k]['level'] != 'NC'}

        def metrics(pred):
            tp = len(pred & actual); fp = len(pred - actual); fn = len(actual - pred)
            rec  = tp / (tp + fn) if (tp + fn) else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            return rec, prec, f1

        ra, pa, fa = metrics(a_pred)
        rb, pb, fb = metrics(b_pred)
        deltas['Recall'].append(rb - ra)
        deltas['Precision'].append(pb - pa)
        deltas['F1'].append(fb - fa)

    return {k: (float(np.mean(v)), np.percentile(v, [2.5, 97.5]))
            for k, v in deltas.items()}


def bootstrap_ci_for(labels_a, labels_b, index, disease):
    """取三项分类指标的 Δ 自助分布；不可用时返回 None。"""
    if not (labels_a and labels_b):
        return None
    la = labels_a.get(index)
    lb = labels_b.get(index)
    dl = labels_a.get(disease) or labels_b.get(disease)
    if not (la and lb and dl):
        return None
    # 两模型的核心分级逐个相同 → 任何重采样下 Δ 都恒为 0，无需跑自助法
    common = set(la) & set(lb)
    if common and all(la[k].get('level') == lb[k].get('level') for k in common):
        return {m: (0.0, (0.0, 0.0)) for m in BOOTSTRAP_METRICS}
    return bootstrap_metric_diff(la, lb, dl)


def _fmt_ci(ci):
    if ci is None:
        return 'N/A'
    _, (lo, hi) = ci
    return f'[{lo:+.3f}, {hi:+.3f}]'


class DiffWindow(QMainWindow):
    def __init__(self, lookup_a, lookup_b, file_a, file_b,
                 diseases, indices,
                 labels_a, labels_b,
                 meta_a, meta_b):
        super().__init__()
        self.setWindowTitle(f'{MODEL_A_NAME} vs {MODEL_B_NAME}')
        self.setMinimumSize(600, 400)
        self.resize(1100, 720)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel(f'{MODEL_A_NAME} vs {MODEL_B_NAME}')
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title.setStyleSheet('color: #2c3e50; padding: 4px;')
        layout.addWidget(title)

        info = QLabel(
            f'A: {os.path.basename(file_a)}    '
            f'B: {os.path.basename(file_b)}'
        )
        info.setStyleSheet('color: #888; padding: 2px;')
        layout.addWidget(info)

        tabs = QTabWidget()

        # ── 原有：指标差异分页 ──
        for idx in indices:
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(6, 6, 6, 6)

            for disease in diseases:
                grp = QGroupBox(disease)
                grp_layout = QVBoxLayout(grp)
                self._build_metric_group(grp_layout, idx, disease,
                                          lookup_a, lookup_b,
                                          labels_a, labels_b)
                tab_layout.addWidget(grp)

            tabs.addTab(tab, idx)

        # ── 空间重叠分页 ──
        if labels_a and labels_b:
            tabs.addTab(self._build_overlap_tab(labels_a, labels_b),
                        '★ Spatial Overlap')

        layout.addWidget(tabs)

        btn = QPushButton('Close')
        btn.setFixedWidth(100)
        btn.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        scroll.setWidget(content)

    # ---------- 原指标差异表 ----------
    def _build_metric_group(self, grp_layout, idx, disease,
                             lookup_a, lookup_b, labels_a, labels_b):
        headers = [''] + [METRIC_LABELS[m] for m in METRIC_KEYS]
        rows = []
        for label, name in [('A', MODEL_A_NAME),
                            ('B', MODEL_B_NAME),
                            ('Δ', 'B-A')]:
            row = [f'{label} ({name})' if name else label]
            for m in METRIC_KEYS:
                key = (idx, disease)
                va = get_val(lookup_a, key, m)
                vb = get_val(lookup_b, key, m)
                if label == 'A':
                    row.append(f'{va:.4f}' if va is not None else '-')
                elif label == 'B':
                    row.append(f'{vb:.4f}' if vb is not None else '-')
                else:
                    if va is not None and vb is not None:
                        d = vb - va
                        row.append((f'{d:+.4f}', d))
                    else:
                        row.append(('-', None))
            rows.append(row)

        # Steiger p / Bootstrap CI：排在 Δ 行的下一行。
        # 相关系数列给 Steiger 检验的双尾 p 值；
        # 分类指标（Recall / Precision / F1）不是相关系数，改给按乡镇
        # 重采样得到的 Δ 的 95% 自助置信区间；
        # Kendall_tau 没有做偏相关也没有自助区间，填 N/A。
        sig_row = ['Steiger p/Bootstrap CI']
        sig_vals = {}
        ci_all = bootstrap_ci_for(labels_a, labels_b, idx, disease)
        for m in METRIC_KEYS:
            if m in STEIGER_METRICS:
                p = steiger_p_for(labels_a, labels_b, idx, disease, m)
                sig_vals[m] = ('p', p)
                sig_row.append(_fmt_p(p))
            elif m in BOOTSTRAP_METRICS:
                ci = (ci_all or {}).get(m)
                sig_vals[m] = ('ci', ci)
                sig_row.append(_fmt_ci(ci))
            else:
                sig_vals[m] = (None, None)
                sig_row.append('N/A')
        rows.append(sig_row)

        tbl = QTableWidget(len(rows), len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.setAlternatingRowColors(True)
        ratios = [3] * len(headers)
        total = sum(ratios)
        for c in range(len(ratios)):
            tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)

        def _resize(e, t=tbl, rt=tuple(ratios), tot=total):
            w = e.size().width()
            for c in range(len(rt)):
                t.setColumnWidth(c, w * rt[c] // tot)
        tbl.resizeEvent = _resize

        for r, row_data in enumerate(rows):
            for c, val in enumerate(row_data):
                if isinstance(val, tuple):
                    text, delta = val
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    if delta is not None:
                        if delta > 0:
                            item.setForeground(QBrush(QColor('#27ae60')))
                        elif delta < 0:
                            item.setForeground(QBrush(QColor('#cc2020')))
                    if r == 2:
                        item.setBackground(QBrush(QColor('#fafafa')))
                    tbl.setItem(r, c, item)
                else:
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(
                        Qt.AlignCenter if c >= 1 else
                        (Qt.AlignLeft | Qt.AlignVCenter)
                    )
                    if r == 2:
                        item.setBackground(QBrush(QColor('#fafafa')))
                    elif r == 3:
                        # Steiger p / Bootstrap CI 行：
                        # 显著项（p<0.05 或 95% CI 不含 0）标红加粗
                        item.setBackground(QBrush(QColor('#f4f6f7')))
                        kind, v = (sig_vals.get(METRIC_KEYS[c - 1])
                                   if c >= 1 else (None, None))
                        if kind == 'p':
                            sig = v is not None and v < 0.05
                        elif kind == 'ci':
                            sig = v is not None and (v[1][0] > 0 or v[1][1] < 0)
                        else:
                            sig = False
                        if sig:
                            item.setForeground(QBrush(QColor('#c0392b')))
                            item.setFont(QFont('Segoe UI', 9, QFont.Bold))
                    tbl.setItem(r, c, item)

        grp_layout.addWidget(tbl)

    # ---------- 空间重叠表 ----------
    def _build_overlap_tab(self, labels_a, labels_b):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(6, 6, 6, 6)

        hint = QLabel(
            f'比较两份代码识别出的「指定级别」乡镇集合是否一致\n'
            f'A = {MODEL_A_NAME}     B = {MODEL_B_NAME}\n'
            'IoU = |A∩B| / |A∪B|   —— 1.0 表示完全重合\n'
            'Dice = 2|A∩B| / (|A|+|B|)   —— 类似 IoU，对大小差异更敏感\n'
            'CoverageA = |A∩B| / |A|   —— A 中该级别的乡镇有多少在 B 中也是该级别\n'
            'CoverageB = |A∩B| / |B|   —— B 中该级别的乡镇有多少在 A 中也是该级别'
        )
        hint.setStyleSheet('color: #555; padding: 4px;')
        hint.setWordWrap(True)
        tab_layout.addWidget(hint)

        # 找出两份都有的指标名
        common_metrics = [m for m in labels_a.keys() if m in labels_b]

        # 只保留 EpiRank —— 其余指标两版结果完全相同，无需比较
        ordered = ['EpiRank'] if 'EpiRank' in common_metrics else []

        headers = ['Level set', '|A|', '|B|', '|A∩B|', '|A∪B|',
                   'IoU', 'Dice', 'CoverageA', 'CoverageB']

        all_rows = []
        for m in ordered:
            for lvl_name, lvl_set in OVERLAP_LEVEL_SETS:
                res = compute_overlap_metrics(labels_a[m], labels_b[m], lvl_set)
                all_rows.append((m, lvl_name, res))

        tbl = QTableWidget(len(all_rows), len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionBehavior(QTableWidget.SelectRows)
        tbl.setAlternatingRowColors(True)

        for r, (metric, lvl_name, res) in enumerate(all_rows):
            iou = res['iou']
            if iou >= 0.9:
                bg = QColor(200, 240, 200)      # 高重合
            elif iou >= 0.6:
                bg = QColor(255, 240, 180)      # 中
            else:
                bg = QColor(255, 210, 210)      # 低

            vals = [
                lvl_name,
                str(res['n_a']), str(res['n_b']),
                str(res['n_i']), str(res['n_u']),
                _fmt(res['iou']), _fmt(res['dice']),
                _fmt(res['cov_a']), _fmt(res['cov_b']),
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if c >= 5:
                    item.setBackground(QBrush(bg))
                elif c == 0:
                    item.setFont(QFont('Segoe UI', 9, QFont.Bold))
                tbl.setItem(r, c, item)

        # ── 等距列宽 ──
        n_cols = len(headers)
        for c in range(n_cols):
            tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)

        def _resize_overlap(e, t=tbl, tot=n_cols):
            w = e.size().width()
            for c in range(tot):
                t.setColumnWidth(c, w // tot)

        tbl.resizeEvent = _resize_overlap
        tab_layout.addWidget(tbl)

        # 明细：以 EpiRank + C-I 级别为例，列出重叠/差异乡镇名称
        if ordered:
            first_metric = ordered[0]
            detail_label = QLabel(
               f'—— C-I 核心的乡镇明细（对应上表第 1 行）'
            )
            detail_label.setStyleSheet(
                'color: #2c3e50; padding: 6px; font-weight: bold;'
            )
            tab_layout.addWidget(detail_label)

            res = compute_overlap_metrics(labels_a[first_metric],
                                            labels_b[first_metric],
                                            {'C-I'})
            only_a = res['set_a'] - res['set_b']
            only_b = res['set_b'] - res['set_a']
            both   = res['inter']

            def _names(labels, keys):
                if not keys:
                    return '(无)'
                parts = []
                for k in sorted(keys, key=lambda x: int(x)):
                    info = labels.get(k, {})
                    name = info.get('town') or info.get('county') or '?'
                    parts.append(name)
                return '、'.join(parts)

            detail_txt = (
                f"两者都是 C-I（{len(both)}）:  "
                f"{_names(labels_a[first_metric], both)}\n\n"
                f"只在 A（{len(only_a)}）:  "
                f"{_names(labels_a[first_metric], only_a)}\n\n"
                f"只在 B（{len(only_b)}）:  "
                f"{_names(labels_b[first_metric], only_b)}"
            )
            detail = QLabel(detail_txt)
            detail.setWordWrap(True)
            detail.setStyleSheet(
                'color: #333; padding: 6px;'
                'background: #f7f7f7; border: 1px solid #ddd;'
            )
            tab_layout.addWidget(detail)

        return tab


def main():
    file_a = find_latest(FOLDER_A, PREFIX_A)
    file_b = find_latest(FOLDER_B, PREFIX_B)
    print(f'A: {os.path.basename(file_a)}')
    print(f'B: {os.path.basename(file_b)}')

    data_a = load_json(file_a)
    data_b = load_json(file_b)

    lookup_a = build_lookup(data_a['summary'])
    lookup_b = build_lookup(data_b['summary'])

    all_keys = set(lookup_a.keys()) | set(lookup_b.keys())
    diseases = sorted({k[1] for k in all_keys})
    indices  = sorted({k[0] for k in all_keys})

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = DiffWindow(
        lookup_a, lookup_b, file_a, file_b,
        diseases, indices,
        data_a['core_labels'], data_b['core_labels'],
        data_a['metadata'], data_b['metadata'],
    )
    window.show()
    app.exec()


if __name__ == '__main__':
    main()