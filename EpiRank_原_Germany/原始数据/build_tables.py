#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 原始数据/ 重建 EpiRank_Germany 使用的 5 个 xlsx 表格，并逐表与现有文件校验。

    python build_tables.py              # 重建到 原始数据/_rebuilt/ 并校验
    python build_tables.py --check      # 只校验，不写任何文件
    python build_tables.py --inplace    # 直接覆盖 EpiRank_Germany 下的原表（危险）

表格 → 原始数据来源
    flu.xlsx        <- flu/cases.csv                                   （ISO 2015-2019 共 261 周求和）
nv.xlsx        <- norovirus_germany_county_2015_2020_raw.csv      （2015-2019 共 5 年发病率 × 人口 / 1e5）
    cn.xlsx         <- OD/trip_count_matrix_..._2024.csv               （4 维聚合为 400×400）
    bs.xlsx         <- OD/Merging_List_Districts.csv + cn.xlsx         （县名/人口为外部来源）
    COVID-19.xlsx   <- SARS-CoV-2-.../Aktuell_...csv                   （本地为 Git-LFS 指针，不可重建）
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

RAW = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RAW)

N_COUNTY = 400
DROPPED_AGS = 16056          # Eisenach：2019/2021 起并入 Wartburgkreis(16061)，400 县方案不含
BERLIN_AGS = 11000
BERLIN_DISTRICTS = range(11001, 11013)   # RKI 把柏林拆成 12 个 "县"
UNKNOWN_AGS = '?????'        # RKI 的 Unbekannt（区县未知）行

# ---- flu/cases.csv 的周窗口 -------------------------------------------------
# 周列没有日期表头，日历锚点由季节形态推定：全序列最高峰 x897 = 2017/18 大流行季
# （峰值 2018 年第 10 周），据此得 x1 = 2001-01-01（周一）、x1036 = 2020-11-02。
# 依据与交叉验证见 数据说明.md 第 2 节。
FLU_X1 = dt.date(2001, 1, 1)         # 第 1 个周列的周一日期
FLU_START = dt.date(2014, 12, 29)    # ISO 2015-W01 的周一
FLU_END = dt.date(2019, 12, 29)      # ISO 2019-W52 的周日（含当日）
FLU_COL0 = (FLU_START - FLU_X1).days // 7        # 0-based 起始列 = 730 → x731
FLU_COL1 = (FLU_END - FLU_X1).days // 7 + 1      # 0-based 结束列 = 991 → x991
FLU_WEEKS = FLU_COL1 - FLU_COL0                  # 261 周 = 5.00 年
assert FLU_WEEKS == 261 and FLU_COL1 <= 1036

NV_YEARS = [2015, 2016, 2017, 2018, 2019]

SHEETS = {
    'bs.xlsx': 'Sheet1',
    'cn.xlsx': '353C',
    'COVID-19.xlsx': '2003',
    'flu.xlsx': '2015_2019',
    'nv.xlsx': '2015_2019',
}


# ---------------------------------------------------------------- 规范县表
def load_canon():
    """400 县的规范行序 = Merging_List_Districts.csv 去掉 16056 后按 AGS 升序。

    这也是 bs/cn/Flu/nv/COVID 五张表共用的行序（db_ID 0..399）。
    """
    m = pd.read_csv(os.path.join(RAW, 'OD', 'Merging_List_Districts.csv'), sep='\t')
    m = m[m['AGS num'] != DROPPED_AGS].sort_values('AGS num').reset_index(drop=True)
    assert len(m) == N_COUNTY, len(m)
    return m


def load_population():
    """flu/population.csv：分号分隔，索引为 AGS（整数），列 2000..2019。"""
    p = pd.read_csv(os.path.join(RAW, 'flu', 'population.csv'), sep=';', index_col=0)
    p.index = p.index.astype(int)
    return p


def rki_ags(df, col='geo_code'):
    """从 RKI 的 geo_code 里取出 5 位 AGS，返回字符串（保留前导零与 '?????'）。"""
    return df[col].str.extract(r'\[([\d?]{5})\]\s*$')[0]


def rki_ags_to_canon(ags_str, canon_index):
    """RKI AGS（字符串）→ 规范表行号。柏林 12 区归并到 11000；未知/不在表内返回 None。"""
    if ags_str is None or ags_str == UNKNOWN_AGS:
        return None
    a = int(ags_str)
    if a in BERLIN_DISTRICTS:
        a = BERLIN_AGS
    return canon_index.get(a)


# ---------------------------------------------------------------- cn.xlsx
def build_cn(canon):
    """2024 年出行矩阵：按 (start_county, end_county) 对 age_group × activity_dest 求和。

    原始 CSV 已经是 400 县 AGS 空间（柏林为 11000，无 16056），直接透视即可。
    """
    od = pd.read_csv(os.path.join(
        RAW, 'OD', 'trip_count_matrix_county_by_age_activity_final_2024.csv'))
    canon_index = {a: i for i, a in enumerate(canon['AGS num'])}
    si = od['start_county'].map(canon_index)
    ei = od['end_county'].map(canon_index)
    assert si.notna().all() and ei.notna().all(), 'OD 矩阵含规范表外的 AGS'
    m = np.zeros((N_COUNTY, N_COUNTY), dtype=np.int64)
    np.add.at(m, (si.to_numpy(), ei.to_numpy()), od['trip_cnt'].to_numpy())
    return m


# ---------------------------------------------------------------- bs.xlsx
def build_bs(canon, cn_matrix, labels):
    """bs.xlsx：坐标来自 Merging_List，通勤四列来自 cn 矩阵，其余为占位常数。

    labels: DataFrame[county, town, population]，这三列**不在 原始数据/ 内**，
            属外部规范名表（见 数据说明.md 第 6 节），需由已有 bs.xlsx 提供。
    """
    bs = pd.DataFrame({
        'db_ID': np.arange(N_COUNTY),
        'county': labels['county'].to_numpy(),
        'town': labels['town'].to_numpy(),
        'area': 1,
        'pos.x': canon['Lon'].to_numpy(),
        'pos.y': canon['Lat'].to_numpy(),
    })
    bs['pos2.x'] = bs['pos.x']
    bs['pos2.y'] = bs['pos.y']
    bs['population'] = labels['population'].to_numpy()
    bs['sub_percentage'] = 1
    bs['sub_area km2'] = 1
    bs['area_km2'] = 1
    bs['pop_den'] = bs['population']
    bs['pop_den (normal)'] = 1
    bs['age 0-14'] = 0.15
    bs['age 15-64'] = 0.65
    bs['age 65+'] = 0.2
    bs['local_commuter_type1'] = np.diag(cn_matrix)
    bs['out_commuter_type1'] = cn_matrix.sum(axis=1)
    bs['in_commuter_type1'] = cn_matrix.sum(axis=0)
    bs['commuter_type1'] = cn_matrix.sum(axis=1)
    bs['railroad_zone'] = 0
    return bs


# ---------------------------------------------------------------- flu.xlsx
def build_flu():
    """flu/cases.csv：401 行（= 400 县 + Eisenach）× 1036 个周列，无表头日期。

    按日历取 ISO 年 2015–2019：x731 … x991 共 261 周（5.00 年），
    即 2014-12-29（2015-W01 周一）… 2019-12-29（2019-W52 周日）。
    含 5 个流感季峰值（2015-W09 / 2016-W11 / 2017-W06 / 2018-W10 / 2019-W08），
    不含 2019/20 季（峰在 2020-W10，已被截掉）。
    行取前 400（与规范行序一致，Eisenach 落在被丢弃的第 401 行）。
    """
    c = pd.read_csv(os.path.join(RAW, 'flu', 'cases.csv'))
    assert c.shape == (N_COUNTY + 1, 1036), c.shape
    return c.iloc[:N_COUNTY, FLU_COL0:FLU_COL1].sum(axis=1).to_numpy(dtype=np.int64)


# ---------------------------------------------------------------- nv.xlsx
def build_nv(canon, pop):
    """norovirus_..._raw.csv：412 地区 × 6 年的**年发病率**（/10 万），不是病例数。

    只取 NV_YEARS（2015–2019 共 5 年），2020 年整年丢弃。
    病例数 = Σ_年 发病率 × 当年人口 / 100000，四舍五入到整数。
    柏林 12 区先取「同一年 12 个区的发病率简单平均」再乘柏林总人口
    （population.csv 只有 11000，没有分区人口）。
    """
    nv = pd.read_csv(os.path.join(RAW, 'norovirus_germany_county_2015_2020_raw.csv'))
    nv = nv[nv['year'].isin(NV_YEARS)].copy()
    canon_index = {a: i for i, a in enumerate(canon['AGS num'])}
    nv['ags'] = rki_ags(nv)
    nv = nv[nv['ags'] != UNKNOWN_AGS].copy()

    is_berlin = nv['ags'].isin({str(a) for a in BERLIN_DISTRICTS})
    berlin = (nv[is_berlin].groupby('year', as_index=False)['incidence'].mean()
              .assign(ags=UNKNOWN_AGS))
    berlin['ags'] = str(BERLIN_AGS)
    nv = pd.concat([nv[~is_berlin], berlin], ignore_index=True)

    nv['row'] = [rki_ags_to_canon(a, canon_index) for a in nv['ags']]
    nv = nv.dropna(subset=['row'])
    # 人口按 AGS 查表；NV_YEARS 全部在 population.csv 的 2000–2019 列内，无回退
    ags_int = nv['ags'].map(lambda s: BERLIN_AGS if int(s) in BERLIN_DISTRICTS else int(s))
    nv['population'] = [
        float(pop.loc[a, str(int(y))])
        for a, y in zip(ags_int, nv['year'])
    ]
    nv['cases'] = nv['incidence'] * nv['population'] / 100000.0

    out = np.zeros(N_COUNTY, dtype=np.float64)
    # 缺失年份（Hof 2016、Kempten 2016/17/19）incidence 为 NaN → 该县-年贡献 0
    np.add.at(out, nv['row'].to_numpy(dtype=int),
              nv['cases'].fillna(0.0).to_numpy())
    return np.rint(out).astype(np.int64)


# ---------------------------------------------------------------- COVID-19
def build_covid(canon):
    """RKI Aktuell_Deutschland_SarsCov2_Infektionen.csv → 400 县累计病例数。

    本地该文件是 134 字节的 Git-LFS 指针（真实体积约 413 MB），无法离线重建。
        需先 `git lfs pull`（见 数据说明.md 第 5 节），再做：
        1. 按 NeuerFall in (0, 1) 过滤，排除 -1（冲销）行；
        2. 按 IdLandkreis 汇总 AnzahlFall；IdLandkreis 为 5 位整数，
           11001..11012（柏林 12 区）先合并成 11000；
        3. 16056（Eisenach）不在 400 县方案内，丢弃；'???' 之类未知行丢弃；
        4. 输出与 bs.xlsx 同序的 (county, town, cases)。
    """
    path = os.path.join(RAW, 'SARS-CoV-2-Infektionen_in_Deutschland-main',
                        'Aktuell_Deutschland_SarsCov2_Infektionen.csv')
    with open(path, 'rb') as f:
        head = f.read(64)
    if head.startswith(b'version https://git-lfs'):
        return None
    df = pd.read_csv(path)
    df = df[df['NeuerFall'].isin([0, 1])]
    canon_index = {a: i for i, a in enumerate(canon['AGS num'])}
    df['row'] = [rki_ags_to_canon(f'{a:05d}', canon_index) for a in df['IdLandkreis']]
    df = df.dropna(subset=['row'])
    out = np.zeros(N_COUNTY, dtype=np.int64)
    np.add.at(out, df['row'].to_numpy(dtype=int), df['AnzahlFall'].to_numpy())
    return out


# ---------------------------------------------------------------- 校验
def _read_target(name):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        return None
    return pd.read_excel(p, sheet_name=SHEETS[name])


def _read_target_cn():
    """cn.xlsx 版式：3 行表头 + 2 行空行 + 400 行；A=seq, B/C=AGS, D/E 空, F..=矩阵。"""
    raw = pd.read_excel(os.path.join(ROOT, 'cn.xlsx'), sheet_name=SHEETS['cn.xlsx'],
                        header=None)
    v = raw.iloc[5:5 + N_COUNTY, 5:5 + N_COUNTY].to_numpy(dtype=np.float64)
    return np.nan_to_num(v, nan=0.0), raw


def _cmp(label, got, want):
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(want, dtype=np.float64)
    if a.shape != b.shape:
        print(f'  {label:12s} FAIL  形状 {a.shape} != {b.shape}')
        return False
    l1 = float(np.abs(a - b).sum())
    tag = 'OK   ' if l1 == 0 else 'DIFF '
    print(f'  {label:12s} {tag} L1={l1:,.0f}  合计 计算={a.sum():,.0f} 现有={b.sum():,.0f}')
    return l1 == 0


def verify(rebuilt):
    print('\n[校验] 重建结果 vs 现有 xlsx')
    results = {}
    # bs.xlsx：整表 22 列
    want = _read_target('bs.xlsx')
    if want is not None and rebuilt['bs.xlsx'] is not None:
        got = rebuilt['bs.xlsx']
        ok = got.shape == want.shape
        results['bs.xlsx'] = ok
        print(f'  {"bs.xlsx":12s} {"OK   " if ok else "FAIL "} 形状 {got.shape} vs {want.shape}')
        if ok:
            num_got = got.select_dtypes('number').to_numpy().ravel()
            num_want = want.select_dtypes('number').to_numpy().ravel()
            results['bs.xlsx'] = _cmp('bs 数值列', num_got, num_want)
            results['bs 文本列'] = bool((got[['county', 'town']].to_numpy() ==
                                        want[['county', 'town']].to_numpy()).all())
            print(f'  {"bs 文本列":12s} '
                  f'{"OK   " if results["bs 文本列"] else "FAIL "} county/town 逐行一致')
    # cn.xlsx：矩阵区块
    if os.path.exists(os.path.join(ROOT, 'cn.xlsx')) and rebuilt['cn.xlsx'] is not None:
        tgt, raw = _read_target_cn()
        results['cn.xlsx'] = _cmp('cn 矩阵', rebuilt['cn.xlsx'].ravel(), tgt.ravel())
        seq_ok = bool(np.array_equal(raw.iloc[5:5 + N_COUNTY, 0].to_numpy(dtype=float),
                                     np.arange(N_COUNTY, dtype=float)))
        print(f'  {"cn 版式":12s} {"OK   " if seq_ok else "FAIL "} '
              f'A 列 seq=0..399, 行块 = 第 6..405 行, 列块 = 第 6..405 列')
        results['cn 版式'] = seq_ok
    # 疾病表：county/town/cases
    for name in ['flu.xlsx', 'nv.xlsx', 'COVID-19.xlsx']:
        want = _read_target(name)
        got = rebuilt[name]
        if got is None:
            print(f'  {name:12s} SKIP  (原始数据不可用，无法重建)')
            results[name] = None
            continue
        names_ok = bool((_read_target('bs.xlsx')['county'].to_numpy()[:N_COUNTY] ==
                         _read_target(name)['county'].to_numpy()).all())
        ok = _cmp(name, np.asarray(got).ravel(), want['cases'].to_numpy())
        print(f'  {"  └行序":12s} {"OK   " if names_ok else "FAIL "} county 与 bs.xlsx 同序')
        results[name] = ok and names_ok
    n_ok = sum(1 for v in results.values() if v is True)
    n_skip = sum(1 for v in results.values() if v is None)
    print(f'\n  => {n_ok} 项完全一致, {len(results)-n_ok-n_skip} 项不一致, {n_skip} 项跳过')
    return results


# ---------------------------------------------------------------- 写出
def _cn_to_frame(canon, cn_matrix):
    """复刻 cn.xlsx 版式：405×405，第 1-3 行 AGS 表头，第 4-5 行空行，
    第 6-405 行为 400 个县：A=seq_no, B=AGS, C=AGS, D/E 空, F..OO=OD 矩阵（0 留空）。
    """
    ags = canon['AGS num'].to_numpy()
    grid = [[None] * (5 + N_COUNTY) for _ in range(3 + 2 + N_COUNTY)]
    for r in range(3):                       # 第 1-3 行：AGS 表头
        for c in range(N_COUNTY):
            grid[r][5 + c] = int(ags[c])
    for i in range(N_COUNTY):                # 第 6-405 行
        row = grid[5 + i]
        row[0] = i
        row[1] = int(ags[i])
        row[2] = int(ags[i])
        for j in range(N_COUNTY):
            if cn_matrix[i, j] > 0:
                row[5 + j] = int(cn_matrix[i, j])
    return pd.DataFrame(grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只校验，不写文件')
    ap.add_argument('--inplace', action='store_true', help='覆盖 ROOT 下的原表')
    ap.add_argument('--out', default=os.path.join(RAW, '_rebuilt'))
    args = ap.parse_args()

    canon = load_canon()
    pop = load_population()
    print(f'[1/5] 规范县表: {N_COUNTY} 行, AGS {canon["AGS num"].iloc[0]}..'
          f'{canon["AGS num"].iloc[-1]}, 已剔除 {DROPPED_AGS}')

    cn_matrix = build_cn(canon)
    print(f'[2/5] cn  : 400x400, 出行 {int((cn_matrix>0).sum())} 条, 总量 {cn_matrix.sum():,}')

    flu = build_flu()
    print(f'[3/5] flu : 合计 {flu.sum():,} '
          f'(ISO 2015-2019, {FLU_WEEKS} 周 x 前 400 行)')

    nv = build_nv(canon, pop)
    print(f'[4/5] nv  : 合计 {nv.sum():,} ({len(NV_YEARS)} 年发病率 x 人口 / 1e5)')

    covid = build_covid(canon)
    print(f'[5/5] COVID: {"不可重建 (本地为 Git-LFS 指针)" if covid is None else covid.sum()}')

    # bs 依赖 cn；county/town/population 为外部名表，从现有 bs.xlsx 取
    bs_path = os.path.join(ROOT, 'bs.xlsx')
    if os.path.exists(bs_path):
        labels = pd.read_excel(bs_path, sheet_name='Sheet1')[['county', 'town', 'population']]
    else:
        raise SystemExit('缺少 bs.xlsx：county/town/population 三列无法从 原始数据/ 推导')
    bs = build_bs(canon, cn_matrix, labels)

    rebuilt = {'bs.xlsx': bs, 'cn.xlsx': cn_matrix, 'COVID-19.xlsx': covid,
               'flu.xlsx': flu, 'nv.xlsx': nv}
    verify(rebuilt)

    if not args.check:
        out = ROOT if args.inplace else args.out
        os.makedirs(out, exist_ok=True)
        for name, built in rebuilt.items():
            if built is None:
                print(f'[写出] 跳过 {name}（无法重建）')
                continue
            if name == 'cn.xlsx':
                df = _cn_to_frame(canon, cn_matrix)
            elif name == 'bs.xlsx':
                df = built
            else:
                df = pd.DataFrame({'county': labels['county'],
                                   'town': labels['town'],
                                   'cases': np.asarray(built)})
            # cn.xlsx 的第 1-3 行本身就是表头，写盘时不能再多写一行 pandas 列名
            df.to_excel(os.path.join(out, name), sheet_name=SHEETS[name],
                        index=False, header=(name != 'cn.xlsx'))
            print(f'[写出] {name} -> {out}')


if __name__ == '__main__':
    main()
