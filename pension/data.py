# -*- coding: utf-8 -*-
"""讀入 PSFD 五波原始資料,轉成分析變數

load_raw()             STEP 1  讀入五波原始檔
build_long(RAW)        STEP 2–3 每一波轉成分析變數,串成長格式(一列 = 一個人 × 一年)
pension_groups(RAW, L) STEP 4  年金分組 → 65 歲以上可分組的樣本 A
add_children(RAW, A)   STEP 5  子女名冊加總回每位長輩
main_source / disease_2024 / panel_pairs / income   其他分析需要的資料"""
import numpy as np
import pandas as pd

from .config import (DATA_DIR, PATHS, BIRTH, HAPPY, HEALTH, LOWINC, ADL, ADL_NAME, ROSTER, SRC, EDU, KID, KID16, FILE_2014, DIS, MAIN_SRC, TH, TH_OTHER,
                     SIX, county)


# ---------- 共用 ----------
def num(f, v):
    """取出一欄並轉成數字;欄位不存在時回傳全部遺漏"""
    return pd.to_numeric(f[v], errors='coerce').astype(float) if (v and v in f.columns) else pd.Series(np.nan, index=f.index)


def pid(f, y):
    """受訪者編號;2024 年起改為 8 碼(舊編號 × 10),除以 10 對回前幾波"""
    x = pd.to_numeric(f['x01'], errors='coerce')
    return ((x / 10).round() if y == 2024 else x).astype('Int64')


def _clean_id(d):
    d = d[d.id.notna()].copy(); d['id'] = d.id.astype('int64')
    return d


# ---------- STEP 1 ----------
def load_raw(log=print):
    RAW = {}
    for y, p in PATHS.items():
        f = DATA_DIR / p
        if not f.exists():
            raise FileNotFoundError(f'找不到 {f}。請依 data/README.md 放入 PSFD 原始檔')
        RAW[y] = pd.read_stata(f, convert_categoricals=False)   # 保留數字代碼,不轉成文字標籤
        log(f'{y}: {RAW[y].shape[0]} 列(受訪者) × {RAW[y].shape[1]} 欄(題目)')
    return RAW


# ---------- STEP 2 ----------
def wave_frame(f, y):
    """把一波問卷的代碼轉成分析變數"""
    d = pd.DataFrame({'id': pid(f, y), 'wave': y})
    d['age'] = y - (num(f, BIRTH[y]) + 1911)                                   # 出生民國年 + 1911 = 西元
    d['male'] = (num(f, 'a01') == 1).astype(float)
    h = num(f, HAPPY[y]); d['happy'] = h.where(h.between(1, 7))                 # 96、98(不知道、拒答)與負數特殊碼排除
    hl = num(f, HEALTH[y]); d['health_bad'] = np.where(hl.between(1, 5), (hl >= 4).astype(float), np.nan)
    li = num(f, LOWINC[y]); d['lowinc_sub'] = np.where(li == 1, 1.0, np.where(li.isin([0, 2]), 0.0, np.nan))
    A10 = pd.concat([num(f, v) for v in ADL[y]], axis=1)                       # 1 很受限、2 有點受限、3 不受限
    valid = A10.isin([1, 2, 3]).all(axis=1)
    d['adl_basic'] = np.where(valid, A10.iloc[:, [0, 1, 5]].isin([1, 2]).any(axis=1).astype(float), np.nan)   # 洗澡穿衣/走一個街口/爬一層樓
    for j, nm in enumerate(ADL_NAME):
        s = A10.iloc[:, j]; d[nm] = np.where(s.isin([1, 2, 3]), s.isin([1, 2]).astype(float), np.nan)
    # 同住名冊:第 3 類 = 配偶;獨居者 2022 f15==0、2024 f15z01==2 會跳過名冊
    R = pd.concat([num(f, f'{ROSTER[y]}{i:02d}') for i in range(1, 35)], axis=1)
    Rv = R.where((R >= 0) & (R <= 20)); rvalid = Rv.notna().sum(axis=1) >= 30
    alone_flag = (num(f, 'f15') == 0) if y == 2022 else (num(f, 'f15z01') == 2) if y == 2024 else pd.Series(False, index=f.index)
    Rv = Rv.where(~alone_flag, 0.0); rvalid = rvalid | alone_flag
    d['alone'] = np.where(rvalid, (Rv.fillna(0).sum(axis=1) == 0).astype(float), np.nan)
    d['co_spouse'] = np.where(rvalid, (Rv.iloc[:, 2] > 0).astype(float), np.nan)
    if y >= 2022:   # 第 6–9 類:已婚兒子、已婚女兒、未婚兒子、未婚女兒(2022、2024 名冊)
        d['co_kid'] = np.where(rvalid, (Rv.iloc[:, 5:9].fillna(0).sum(axis=1) > 0).astype(float), np.nan)
    # 經濟來源(複選,2018 起):1、2 工作;3、4 自己 / 配偶退休金;5、6 儲蓄;7 子女;9 政府津貼
    if y in SRC:
        fmt = SRC[y]
        answered = pd.concat([num(f, fmt.format(i)) for i in (1, 3, 7)], axis=1).isin([0, 1, 2]).any(axis=1)
        for name, codes in [('src_work', (1, 2)), ('src_pension', (3, 4)), ('src_saving', (5, 6)), ('src_child', (7,)), ('src_gov', (9,))]:
            d[name] = np.where(answered, pd.concat([num(f, fmt.format(i)) == 1 for i in codes], axis=1).any(axis=1).astype(float), np.nan)
    ed = num(f, EDU[y])
    d['edu_g'] = pd.cut(ed.where(ed.between(1, 15)), [0, 3, 6, 11, 15], labels=['國小以下', '國初中', '高中職專科', '大學以上']).astype(object)
    d['zip'] = num(f, 'x05').where(num(f, 'x05').between(100, 983))
    d['county'] = d.zip.map(county)
    d['six'] = np.where(d.county.notna(), d.county.isin(SIX).astype(float), np.nan)
    if y >= 2022:   # 只有 2022、2024 年問的題目
        k33 = num(f, 'k33')    # 您(和您配偶)在應付每個月的生活費用上,是否足夠或有困難? 1 足夠 ~ 4 相當困難
        d['hard'] = np.where(k33.between(1, 4), (k33 >= 3).astype(float), np.nan)
        s = num(f, 'm02z10'); d['lonely'] = np.where(s.between(1, 4), (s >= 3).astype(float), np.nan)   # 經常 / 幾乎總是寂寞
    else:
        f01 = num(f, 'f01'); d['sick_rely'] = np.where(f01.between(1, 5), (f01 <= 2).astype(float), np.nan)  # 生病時可依靠家人
    return d.drop_duplicates('id')


# ---------- STEP 3 ----------
def build_long(RAW, log=print):
    L = pd.concat([wave_frame(f, y) for y, f in RAW.items()], ignore_index=True)
    L = _clean_id(L); L['female'] = 1 - L.male
    log(f'長格式資料:{len(L)} 列、{L.id.nunique()} 人')
    return L


# ---------- STEP 4 ----------
def retirement_history(RAW):
    """是否辦過退休。2018、2020 直接問;2022 起只問「上次訪問後有沒有新辦」,
    所以要往前累積:只要說過一次「有」,之後都算有(cummax)"""
    hist = []
    for y in [2018, 2020, 2022, 2024]:
        f = RAW[y]
        if y <= 2020:
            c05 = num(f, 'c05')                                    # 是否辦過退休:1、2 有;3 沒有
            ever = np.where(c05.isin([1, 2]), 1.0, np.where(c05 == 3, 0.0, np.nan))
        else:
            ever = np.where(num(f, 'k07r1').isin([1, 2]), 1.0, np.nan)
        hist.append(pd.DataFrame({'id': pid(f, y), 'wave': y, 'ever_now': ever}).drop_duplicates('id'))
    H = _clean_id(pd.concat(hist)).sort_values(['id', 'wave'])
    ever_yes = H.groupby('id').ever_now.transform(lambda s: (s == 1).cummax())
    known_never = H.groupby('id').ever_now.transform(lambda s: (s == 0).cummax())
    H['ever'] = np.where(ever_yes, 1.0, np.where(known_never, 0.0, np.nan))
    return H[['id', 'wave', 'ever']]


def _grp(r):
    if pd.isna(r.src_pension): return np.nan
    if r.src_pension == 1: return 'A 月領年金'
    if r.ever == 1: return 'B 一次領完型'
    if r.ever == 0: return 'CD 制度外型'
    return np.nan


def pension_groups(RAW, L, log=print):
    """月領年金 vs 沒有月領年金(含一次領完、從未辦過退休)。回傳 (M 全部年齡, A 65 歲以上可分組)"""
    M = L.merge(retirement_history(RAW), on=['id', 'wave'], how='left')
    M['grp3'] = M.apply(_grp, axis=1)
    A = M[(M.age >= 65) & M.grp3.notna()].copy()
    A['nop'] = (A.grp3 != 'A 月領年金').astype(float)
    log(f'65 歲以上、可分組:{len(A)} 人次、{A.id.nunique()} 人;至少兩波 {(A.groupby("id").size() >= 2).mean()*100:.0f}%')
    return M, A


# ---------- STEP 5 ----------
def add_children(RAW, A, log=print):
    """子女名冊:每個子女一列 → 加總回每位長輩。回傳 (A 加上子女欄位, K 子女層級資料)"""
    kid_rows = []
    for y, v in KID.items():
        f = RAW[y]; x = pid(f, y); page = y - (num(f, BIRTH[y]) + 1911)
        for k in range(1, 13):
            if v['gave'].format(k) not in f.columns: break
            g = num(f, v['gave'].format(k)); amt = num(f, v['amt'].format(k)); wk = num(f, v['work'].format(k)); ed = num(f, v['edu'].format(k))
            d = pd.DataFrame({'id': x, 'wave': y, 'page': page, 'kid': k,
                              'gave': np.where(g == 1, 1.0, np.where(g == 2, 0.0, np.nan)),       # 有給錢 1、沒給 2
                              'amt': amt.where((amt >= 0) & (amt < 500000)),
                              'k_full': np.where(wk.isin([1, 2, 3]), (wk == 1).astype(float), np.nan),   # 1 全職
                              'k_college': np.where(ed.between(1, 15), (ed >= 12).astype(float), np.nan)})
            kid_rows.append(d[d.gave.notna() & (d.page >= 65)])
    KD = _clean_id(pd.concat(kid_rows, ignore_index=True))
    KD['amt_g'] = np.where(KD.gave == 1, KD.amt, 0.0)                               # 沒給錢 → 0
    kp = KD.groupby(['id', 'wave']).agg(kid_total=('amt_g', lambda s: s.sum(min_count=1)),   # 全部遺漏才是 NaN
                                        any_full=('k_full', 'max'), n_full=('k_full', 'sum')).reset_index()
    A = A.merge(kp, on=['id', 'wave'], how='left')
    K = KD.merge(A[['id', 'wave', 'nop']], on=['id', 'wave'])
    log(f'子女資料:{len(K)} 筆(子女 × 年份);子女合計每月給的中位數 有年金 {A[A.nop==0].kid_total.median():.0f} / 沒有年金 {A[A.nop==1].kid_total.median():.0f} 元;'
        f'子女大學以上 有年金 {K[K.nop==0].k_college.mean()*100:.0f}% / 沒有年金 {K[K.nop==1].k_college.mean()*100:.0f}%')
    return A, K


def children_all(RAW, L):
    """所有 65 歲以上長輩(不限年金分組)2016–2024 年子女給的錢,用來檢驗「有子女就有人養」。
    回傳 (KA 子女層級, PA 長輩層級:子女數、合計金額、居住縣市當年的最低生活費)"""
    spec = {2016: KID16, **{y: v for y, v in KID.items()}}
    rows = []
    for y, v in spec.items():
        f = RAW[y]; x = pid(f, y); page = y - (num(f, BIRTH[y]) + 1911)
        for k in range(1, 13):
            if v['gave'].format(k) not in f.columns: break
            g, amt = num(f, v['gave'].format(k)), num(f, v['amt'].format(k))
            d = pd.DataFrame({'id': x, 'wave': y, 'page': page, 'kid': k,
                              'gave': np.where(g == 1, 1.0, np.where(g == 2, 0.0, np.nan)),
                              'amt': amt.where((amt >= 0) & (amt < 500000))})
            rows.append(d[d.gave.notna() & (d.page >= 65)])
    KA = _clean_id(pd.concat(rows, ignore_index=True))
    KA['amt_g'] = np.where(KA.gave == 1, KA.amt, 0.0)
    PA = KA.groupby(['id', 'wave']).agg(n_kid=('kid', 'size'), kid_total=('amt_g', lambda s: s.sum(min_count=1))).reset_index()
    PA = PA.merge(L[['id', 'wave', 'county', 'female', 'age']], on=['id', 'wave'], how='left')
    p14 = _wave2014()
    if p14 is not None:
        PA = pd.concat([p14, PA], ignore_index=True)
    PA['th'] = [TH[w].get(c, TH_OTHER[w]) for w, c in zip(PA.wave, PA.county)]
    return KA, PA


def _wave2014():
    """2014 年(若 data/ 裡有):所有子女合計每月給多少;子女數由子女名冊計算。沒有這個檔案時回傳 None"""
    hits = sorted(DATA_DIR.rglob(FILE_2014))
    if not hits:
        return None
    f = pd.read_stata(hits[0], convert_categoricals=False)
    age = 2014 - (num(f, 'a02a') + 1911)
    g, a = num(f, 'b15a01'), num(f, 'b15a02')
    tot = np.where(g == 2, 0.0, np.where((g == 1) & (a < 9999990), a, np.nan))     # 9999995–9999999 為不知道、拒答等
    nk = pd.concat([num(f, f'b14bc{k}').isin([1, 2]) for k in range(1, 7)], axis=1).sum(axis=1)
    z = num(f, 'x05').where(num(f, 'x05').between(100, 983))
    d = pd.DataFrame({'id': pd.to_numeric(f.x01, errors='coerce'), 'wave': 2014, 'n_kid': nk, 'kid_total': tot,
                      'county': z.map(county), 'female': (num(f, 'a01') == 2).astype(float), 'age': age})
    return _clean_id(d[(d.age >= 65) & (d.n_kid > 0)])


# ---------- 其他 ----------
def main_source(RAW, A):
    """2022、2024 最主要經濟來源(k21z01);只有單一來源者會跳答,用複選題補"""
    ms = pd.concat([pd.DataFrame({'id': pid(RAW[y], y), 'wave': y, 'main': num(RAW[y], 'k21z01').map(MAIN_SRC)}) for y in [2022, 2024]])
    W2 = A[A.wave >= 2022].merge(_clean_id(ms), on=['id', 'wave'])
    srcmap = {'src_work': '自己或配偶工作', 'src_pension': '退休金', 'src_saving': '儲蓄利息租金投資', 'src_child': '子女奉養', 'src_gov': '政府救助或津貼'}
    single = W2.main.isna()
    for c, lab in srcmap.items():
        only = single & (W2[c] == 1) & (W2[[k for k in srcmap if k != c]].fillna(0).sum(axis=1) == 0)
        W2.loc[only, 'main'] = lab
    return W2


def disease_2024(RAW, A):
    """2024 年 13 種慢性病(0/1)"""
    f24 = RAW[2024]; dz = pd.DataFrame({'id': pid(f24, 2024), 'wave': 2024})
    for v, nm in DIS.items():
        s = num(f24, v); dz[nm] = np.where(s.isin([0, 1]), s, np.nan)
    return A[A.wave == 2024].merge(_clean_id(dz), on=['id', 'wave'])


SHOCKS = ['sh_health', 'sh_adl', 'sh_work', 'sh_child', 'sh_spouse']


def panel_pairs(M):
    """相鄰兩波都受訪、第一波 60 歲以上的配對;標記兩年內遇到的衝擊"""
    Wf = {y: M[M.wave == y].drop_duplicates('id').set_index('id') for y in PATHS}
    rows = []
    for y0 in [2016, 2018, 2020, 2022]:
        a, b = Wf[y0], Wf[y0 + 2]
        ids = a.index.intersection(b.index); a, b = a.loc[ids], b.loc[ids]           # 兩波都受訪的人
        g0 = a['grp3'] if y0 >= 2018 else Wf[2018]['grp3'].reindex(ids)            # 2016 年沒有年金題,用 2018 年的分組
        t = pd.DataFrame({'y0': y0, 'age0': a.age, 'female': 1 - a.male, 'grp': g0})
        for c in ['happy', 'health_bad', 'adl_basic', 'src_work', 'src_child', 'co_spouse']:
            t[c + '0'] = a[c] if c in a else np.nan; t[c + '1'] = b[c] if c in b else np.nan
        rows.append(t[t.age0 >= 60])
    T = pd.concat(rows).reset_index(); T = T[T.grp.notna()].copy()
    both = lambda c: T[c + '0'].notna() & T[c + '1'].notna()
    SH = {'sh_health': ((T.health_bad0 == 0) & (T.health_bad1 == 1)).where(both('health_bad')),   # 健康變差
          'sh_adl': ((T.adl_basic0 == 0) & (T.adl_basic1 == 1)).where(both('adl_basic')),        # 新出現行動受限
          'sh_work': ((T.src_work0 == 1) & (T.src_work1 == 0)).where(both('src_work')),          # 失去工作收入
          'sh_child': ((T.src_child0 == 1) & (T.src_child1 == 0)).where(both('src_child')),      # 失去子女奉養
          'sh_spouse': ((T.co_spouse0 == 1) & (T.co_spouse1 == 0)).where(both('co_spouse'))}     # 不再與配偶同住
    for k, v in SH.items(): T[k] = v.astype(float)
    T['any_shock'] = (T[SHOCKS].fillna(0).sum(axis=1) > 0).astype(float)
    T['nop'] = (T.grp != 'A 月領年金').astype(float)
    T['d_happy'] = T.happy1 - T.happy0
    return T, Wf


def income(RAW):
    """2022、2024 收入與津貼:本人與配偶的工作收入、國民年金 / 老農津貼、全家總收入"""
    add = []
    for y, hh_inc, hh_n in [(2022, 'k22', 'k24z04'), (2024, 'k24e', 'k24a')]:
        f = RAW[y]
        wr1, wr2, a1, a2 = num(f, 'k04z02r1'), num(f, 'k04z02r2'), num(f, 'k18r1'), num(f, 'k18r2')
        hi, hn = num(f, hh_inc), num(f, hh_n)
        add.append(pd.DataFrame({'id': pid(f, y), 'wave': y,
            'work_m': wr1.where((wr1 >= 0) & (wr1 < 2e7)).fillna(0) / 12,            # 本人去年工作收入 ÷ 12
            'sp_work_m': wr2.where((wr2 >= 0) & (wr2 < 2e7)).fillna(0) / 12,         # 配偶
            'np': num(f, 'k17z01r1').where(num(f, 'k17z01r1').isin([0, 1])),         # 本人有沒有領國民年金
            'farm': num(f, 'k17z02r1').where(num(f, 'k17z02r1').isin([0, 1])),       # 本人有沒有領老農津貼
            'ben_amt': a1.where((a1 > 0) & (a1 < 100000)),                           # 每月一共多少
            'sp_np': num(f, 'k17z01r2').where(num(f, 'k17z01r2').isin([0, 1])),
            'sp_farm': num(f, 'k17z02r2').where(num(f, 'k17z02r2').isin([0, 1])),
            'sp_amt': a2.where((a2 > 0) & (a2 < 100000)),
            'hh_inc_m': hi.where((hi >= 0) & (hi < 1e8)) / 12, 'hh_n': hn.where((hn >= 1) & (hn <= 30))}).drop_duplicates('id'))
    return _clean_id(pd.concat(add))
