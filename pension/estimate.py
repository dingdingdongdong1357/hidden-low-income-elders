# -*- coding: utf-8 -*-
"""困難長輩、每月缺口、全台推估與拔靴法信賴區間(STEP 9–11)

困難長輩 = 沒有按月年金 + 應付生活費有困難(need);其中沒有領中低收入戶生活補助者 = 沒被接住(sand)
缺口     = max(0, 居住縣市最低生活費 − 每人每月可用的錢)
推估     = 65 歲以上人口 × 加權後的比例;總額 = 人數 × 平均缺口 × 12"""
import numpy as np
import pandas as pd

from .config import TH, TH_OTHER, STAT, POPC, POP, MOHW, N_BOOT, SEED


def sandwich_frame(A, AD):
    """2022、2024 年 65 歲以上、生活費與補助都有回答的人,標記困難長輩與其中沒被接住者"""
    B = A[A.wave >= 2022].copy()
    D = B.merge(AD, on=['id', 'wave'], how='left').dropna(subset=['hard', 'lowinc_sub']).copy()
    D['need'] = ((D.nop == 1) & (D.hard == 1)).astype(float)                     # 困難長輩
    D['sand'] = ((D.need == 1) & (D.lowinc_sub == 0)).astype(float)               # 其中沒被救助接住
    return D


def benefit(npv, farm, amt):
    """本人領哪一種津貼、每月多少(有回報金額用回報值,沒有就代入法定金額)"""
    t = np.select([farm == 1, npv == 1, (npv == 0) & (farm == 0)], ['老農津貼', '國民年金', '兩者皆無'], '未回答')
    b = pd.Series(amt).where(pd.Series(t).isin(list(STAT)).values).fillna(pd.Series(t).map({**STAT, '兩者皆無': 0}))
    return t, b.values


def add_gap(D):
    """每人每月可用的錢 = (子女給的 + 本人工作 + 本人津貼 [+ 配偶工作 + 配偶津貼]) ÷ (1 或 2 人)"""
    D = D.copy()
    D['th'] = [TH[w].get(c, TH_OTHER[w]) for w, c in zip(D.wave, D.county)]
    D['type'], D['ben'] = benefit(D.np.values, D.farm.values, D.ben_amt.values)
    _, sp_b = benefit(D.sp_np.values, D.sp_farm.values, D.sp_amt.values)
    couple = (D.co_spouse == 1).astype(int)
    D['sp_ben'] = np.where(couple == 1, pd.Series(sp_b).fillna(0).values, 0)
    res = D.kid_total.fillna(0) + D.work_m.fillna(0) + D.ben + np.where(couple == 1, D.sp_work_m.fillna(0) + D.sp_ben, 0)
    D['pc_res'] = res / (1 + couple)
    D['gap'] = (D.th - D.pc_res).clip(lower=0)                                        # 超過最低生活費者缺口為 0
    D.loc[D.type == '未回答', 'gap'] = np.nan
    return D


def funnel(D, J, log=print):
    """從所有長輩一步步收窄(加權,%):子女給的錢不到最低生活費 → 沒有按月年金 → 付不出生活費 → 沒被救助接住"""
    d = D[D.kid_total.notna()].copy(); w = weights(d)
    pct = lambda m: round(float(np.average(m.astype(float), weights=w)) * 100, 1)
    s1 = d.kid_total < d.th; s2 = s1 & (d.nop == 1); s3 = s2 & (d.hard == 1); s4 = s3 & (d.lowinc_sub == 0)
    within = lambda m: round(float(np.average(d.hard[m], weights=w[m])) * 100, 1)
    hard = d.hard == 1
    J['funnel'] = {'n': len(d), '子女給的錢不到最低生活費': pct(s1), '且沒有按月年金': pct(s2), '且付不出生活費': pct(s3), '且沒被救助接住': pct(s4),
                   '子女給不夠者付不出_有年金': within(s1 & (d.nop == 0)), '子女給不夠者付不出_沒年金': within(s1 & (d.nop == 1)),
                   '沒年金且有困難者領到補助': round(float(np.average(d.lowinc_sub[hard & (d.nop == 1)], weights=w[hard & (d.nop == 1)])) * 100, 1)}
    log('漏斗(加權 %):', J['funnel'])
    # 敏感度:嚴格定義(自評有困難,且每人可用的錢也低於最低生活費)
    st = (D.need == 1) & (D.gap > 0); w2 = weights(D); share = float(np.average(st, weights=w2)); g = float(np.average(D.gap[st], weights=w2[st]))
    J['strict'] = {'人次': int(st.sum()), '佔65+(%)': round(share * 100, 1), '人數(萬)': round(POP * share, 1), '平均缺口': round(g), '每年總額(億)': round(POP * share * 1e4 * g * 12 / 1e8)}
    log('嚴格定義:', J['strict'])


def income_profile(D, J, log=print):
    """圖 2:100 位長輩依「子女給的錢夠不夠 × 付不出生活費」拆解,以及子女錢不夠者中「過得去 vs 付不出」的其他收入來源(加權 %)"""
    d = D[D.kid_total.notna()].copy(); d['w'] = weights(d)
    d['pension'] = (d.nop == 0).astype(float)
    low, hard, nop, sub = d.kid_total < d.th, d.hard == 1, d.nop == 1, d.lowinc_sub == 1
    share = lambda m: round(float(d.w[m].sum() / d.w.sum()) * 100, 1)
    tree = {'子女給的錢夠': share(~low), '子女給的錢夠_付不出': share(~low & hard),
            '子女給的錢不夠': share(low), '子女給的錢不夠_過得去': share(low & ~hard), '子女給的錢不夠_付不出': share(low & hard),
            '付不出_有按月年金': share(low & hard & ~nop), '付不出_沒有按月年金': share(low & hard & nop),
            '付不出_沒有按月年金_領補助': share(low & hard & nop & sub), '付不出_沒有按月年金_沒領補助': share(low & hard & nop & ~sub),
            '子女給的錢夠_付不出_沒年金_沒領補助': share(~low & hard & nop & ~sub)}
    items = {'pension': '按月年金', 'src_saving': '儲蓄、利息、租金', 'src_work': '自己或配偶工作', 'src_gov': '政府津貼(國保、老農等)'}
    prof = {}
    for lab, m in [('過得去', low & ~hard), ('付不出', low & hard)]:
        g = d[m]
        prof[lab] = {'n': int(m.sum()), **{nm: round(float(np.average(g[c].dropna(), weights=g.w[g[c].notna()])) * 100, 1) for c, nm in items.items()}}
    J['income_profile'] = {'tree': tree, 'profile': prof}
    log('100 位長輩的拆解(加權 %):', tree)
    log('子女錢不夠者的其他收入(加權 %):', prof)


def weights(d):
    """事後加權:權數 = 全國比例 ÷ 樣本比例(年齡 65–74 / 75+ × 性別,共 4 格)"""
    key = list(zip(d.female.astype(int), (d.age >= 75).astype(int)))
    share = pd.Series(key).value_counts(normalize=True); tot = sum(POPC.values())
    return np.array([POPC[k] / tot / share[k] for k in key])


def stats_of(d, w):
    """由同一份樣本、同一組權數算出所有推估值(比例、人數、平均缺口、總額)"""
    s = d.need.values; hmask = d.hard.values == 1; m = (s == 1) & d.gap.notna().values
    share_a = np.average(s, weights=w); share_b = np.average(s[hmask], weights=w[hmask]); gap = np.average(d.gap.values[m], weights=w[m])
    share_s = np.average(d.sand.values, weights=w)
    return {'困難長輩佔65+(%)': share_a * 100, '方法A人數(萬)': POP * share_a, '方法B人數(萬)': MOHW * share_b,
            '其中沒被接住(萬)': POP * share_s, '平均每月缺口(元)': gap, '每年總金額(億)': POP * share_a * 1e4 * gap * 12 / 1e8}


ALLOW = (8329, 4164)       # 中低收入老人生活津貼(2024 起):未達 1.5 倍 / 1.5–2.5 倍最低生活費


def hidden_poor(d, w, drop=None):
    """被隱藏的中低收入戶(沒有按月年金的長輩;子女只計實際給付;津貼依現行規定只能擇一)
    y = 每人每月可用的錢(pc_res);A = 應領津貼;g = 擇一後實際增加的錢
    第一階段 C1:符合資格卻沒領的人都發津貼;第二階段 C2:所有人領完津貼後仍低於最低生活費的差額
    drop:穩健性檢查,被排除者一律視為不符合資格"""
    P0 = (d.nop == 1).values & d.gap.notna().values
    S = (d.lowinc_sub == 1).values
    ok = np.ones(len(d), bool) if drop is None else ~drop(d)
    y, L_ = d.pc_res.values, d.th.values
    A_ = np.where(y < 1.5 * L_, ALLOW[0], np.where(y < 2.5 * L_, ALLOW[1], 0))
    g = np.maximum(A_ - np.where(d.type.isin(['老農津貼', '國民年金']), d.ben, 0), 0)
    H = P0 & ~S & (A_ > 0) & ok
    G = np.where(P0 & ok, np.maximum(L_ - y - np.where(A_ > 0, g, 0), 0), 0)
    return {'隱藏中低收入(萬)': POP * np.average(H, weights=w), '每人每月增加(元)': np.average(g[H], weights=w[H]) if H.any() else 0.0,
            '第一階段(億)': POP * 1e4 * np.average(np.where(H, g, 0), weights=w) * 12 / 1e8,
            '仍不足(萬)': POP * np.average(G > 0, weights=w), '第二階段(億)': POP * 1e4 * np.average(G, weights=w) * 12 / 1e8}


DROPS = {'主要結果': None,
         '排除有存款利息收入者': lambda d: (d.src_saving == 1).values}   # 有統計依據的篩選(見 hidden_summary 的檢驗)


def hidden_summary(D, J, n_boot=N_BOOT, log=print):
    """描述:沒有按月年金者的本人收入與資格;兩階段成本與穩健性檢查(拔靴法信賴區間)"""
    w = weights(D); P0 = (D.nop == 1) & D.gap.notna()
    couple = (D.co_spouse == 1).astype(int)
    own = (D.work_m.fillna(0) + D.ben + np.where(couple == 1, D.sp_work_m.fillna(0) + D.sp_ben, 0)) / (1 + couple)
    pw = w[P0.values]; pct = lambda m: round(float(np.average(m[P0].astype(float), weights=pw)) * 100, 1)
    desc = {'沒年金長輩(萬)': round(POP * float(np.average(P0, weights=w)), 1), '本人收入中位數': float(own[P0].median()),
            '不算子女_低於最低生活費': pct(own < D.th), '不算子女_低於1.5倍': pct(own < 1.5 * D.th),
            '只計實際給付_低於1.5倍': pct(D.pc_res < 1.5 * D.th), '只計實際給付_低於2.5倍': pct(D.pc_res < 2.5 * D.th),
            '已領補助': pct(D.lowinc_sub == 1), '有存款利息收入': pct(D.src_saving == 1), '與子女同住': pct(D.co_kid == 1),
            '與子女同住_付不出': round(float(np.average(D.hard[P0 & (D.co_kid == 1)], weights=w[(P0 & (D.co_kid == 1)).values])) * 100, 1),
            '未與子女同住_付不出': round(float(np.average(D.hard[P0 & (D.co_kid == 0)], weights=w[(P0 & (D.co_kid == 0)).values])) * 100, 1)}
    J['hidden'] = {'desc': desc, 'cost': {}}
    log('沒有按月年金的長輩:', desc)
    # 篩選的統計依據:依收入符合資格者中,有存款利息收入者是否比較不缺錢(邏輯斯迴歸,控制年齡、性別、年份)
    import statsmodels.formula.api as smf
    from .models import clus
    A_ = np.where(D.pc_res < 1.5 * D.th, 1, np.where(D.pc_res < 2.5 * D.th, 1, 0))
    Hd = D[P0 & (D.lowinc_sub == 0) & (A_ == 1)].copy()
    m = smf.logit('hard ~ src_saving + age + female + C(wave)', data=Hd).fit(disp=0, **clus(Hd))
    a, b = Hd[Hd.src_saving == 1], Hd[Hd.src_saving == 0]
    J['hidden']['screen_test'] = {'有存款利息_付不出(%)': round(float(a.hard.mean()) * 100, 1), 'n有': len(a),
                                  '沒有_付不出(%)': round(float(b.hard.mean()) * 100, 1), 'n沒有': len(b),
                                  'OR': round(float(np.exp(m.params['src_saving'])), 2), 'p': round(float(m.pvalues['src_saving']), 4)}
    log('篩選依據:', J['hidden']['screen_test'])
    keep = None
    for lab, dr in DROPS.items():
        fn = lambda d, w, dr=dr: hidden_poor(d, w, dr)
        est = fn(D, w); Bt = bootstrap_fn(D, fn, n_boot); c = ci(Bt)
        if keep is None: keep = (Bt, est)                                    # 主要結果的重抽,畫拔靴法圖用
        J['hidden']['cost'][lab] = {k: [round(float(v), 2), round(c[k][0], 2), round(c[k][1], 2)] for k, v in est.items()}
        log(f'  {lab}:' + ';'.join(f'{k} {v:.0f}(95% CI {c[k][0]:.0f}–{c[k][1]:.0f})' for k, v in est.items()))
    J['hidden']['比例與每人金額的相關'] = round(float(keep[0]['隱藏中低收入(萬)'].corr(keep[0]['每人每月增加(元)'])), 3)
    return keep


def bootstrap_fn(D, fn, n_boot=N_BOOT, seed=SEED):
    """同 bootstrap(),但每次重抽改算 fn(樣本, 權數)"""
    psu_rows, strata = design(D)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(n_boot):
        parts = []
        for st, ps in strata.items():
            parts += [psu_rows[p] for p in (ps if len(ps) == 1 else rng.choice(ps, len(ps)))]
        b = pd.concat(parts, ignore_index=True)
        boot.append(fn(b, weights(b)))
    return pd.DataFrame(boot)


def design(D):
    """抽樣設計:分層 = 縣市;抽樣單位(PSU)= 鄉鎮市區(以郵遞區號代表)。
    同一人多年都用第一次出現的地址,確保同一人永遠在同一個 PSU"""
    D = D.copy()
    D['psu'] = D.groupby('id').zip.transform('first').fillna(-1)
    D['stratum'] = D.groupby('id').county.transform('first').fillna('未知')
    psu_rows = {p: g for p, g in D.groupby('psu')}
    strata = D.drop_duplicates('psu').groupby('stratum').psu.apply(list).to_dict()
    return psu_rows, strata


def bootstrap(D, n_boot=N_BOOT, seed=SEED):
    """分層群集拔靴法:每一次在每個縣市內「有放回地」重抽同樣數目的鄉鎮,整個鄉鎮的人一起帶進來,
    重算權數,再用同一份重抽樣本同時算出比例、缺口與總額。回傳 n_boot 列的 DataFrame"""
    psu_rows, strata = design(D)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(n_boot):
        parts = []
        for st, ps in strata.items():
            parts += [psu_rows[p] for p in (ps if len(ps) == 1 else rng.choice(ps, len(ps)))]
        b = pd.concat(parts, ignore_index=True)
        boot.append(stats_of(b, weights(b)))
    return pd.DataFrame(boot)


def ci(Bt, level=0.95):
    """百分位法信賴區間"""
    a = (1 - level) / 2
    return {k: (float(Bt[k].quantile(a)), float(Bt[k].quantile(1 - a))) for k in Bt.columns}
