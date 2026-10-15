# -*- coding: utf-8 -*-
"""報告第 3 節的比較分析(STEP 6–8、12)。每個函式把結果印到 log,並把畫圖用的數字寫進 J"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import ADL_NAME, DIS, TH_OTHER
from .data import SHOCKS
from .models import clus, adjusted


# ---------- STEP 5b:檢驗「有子女就有人養」 ----------
def child_trend(KA, PA, J, log=print):
    """所有 65 歲以上長輩,每年:每位子女給多少、子女數、合計多少、達到當年居住縣市最低生活費的比例(未加權)"""
    rows = {}
    for y in sorted(PA.wave.unique()):
        k, p = KA[KA.wave == y], PA[(PA.wave == y) & PA.kid_total.notna()]
        per_kid = len(k) > 0                                          # 2014 年只問合計,沒有逐位子女的資料
        rows[int(y)] = {'子女數': len(k), '有給比例': round(float(k.gave.mean()) * 100, 1) if per_kid else None,
                        '每位子女中位數': float(k.amt_g.median()) if per_kid else None, '每位子女平均': round(float(k.amt_g.mean())) if per_kid else None,
                        '長輩數': len(p), '平均子女數': round(float(p.n_kid.mean()), 2),
                        '每位長輩合計中位數': float(p.kid_total.median()), '每位長輩合計平均': round(float(p.kid_total.mean())),
                        '完全沒拿(%)': round(float((p.kid_total == 0).mean()) * 100, 1),
                        '達最低生活費(%)': round(float((p.kid_total >= p.th).mean()) * 100, 1),
                        '最低生活費(臺灣)': int(TH_OTHER[int(y)])}
    J['child_trend'] = rows
    log('子女給的錢(所有 65 歲以上,未加權):')
    log(pd.DataFrame(rows).T.to_string())


def full_model(A, K, J, log=print):
    """2022、2024:同時控制健康、儲蓄、子女、居住安排、教育、地區後,沒有按月年金與生活費困難的關係"""
    nk = K.groupby(['id', 'wave']).size().rename('n_kid').reset_index()
    B = A[A.wave >= 2022].merge(nk, on=['id', 'wave'], how='left')
    B['kid_k'] = B.kid_total / 1000
    B['primary'] = (B.edu_g == '國小以下').astype(float).where(B.edu_g.notna())
    B['college'] = (B.edu_g == '大學以上').astype(float).where(B.edu_g.notna())
    X = ['nop', 'kid_k', 'n_kid', 'any_full', 'primary', 'college', 'co_spouse', 'alone', 'adl_basic', 'health_bad', 'six', 'src_work', 'src_saving']
    d = B.dropna(subset=['hard', 'age', 'female'] + X).copy()
    f = 'hard ~ ' + ' + '.join(['age', 'female', 'C(wave)'] + X)
    m = smf.logit(f, data=d).fit(disp=0, **clus(d))
    mn = smf.logit(f, data=d).fit(disp=0)
    ame = mn.get_margeff(at='overall').summary_frame()
    cis = np.exp(m.conf_int())
    res = {v: {'OR': round(float(np.exp(m.params[v])), 2), 'p': round(float(m.pvalues[v]), 4), 'AME': round(float(ame.loc[v, 'dy/dx']) * 100, 1),
               'CI': [round(float(cis.loc[v, 0]), 2), round(float(cis.loc[v, 1]), 2)],
               'R2_drop': round(float(mn.prsquared - smf.logit('hard ~ ' + ' + '.join(['age', 'female', 'C(wave)'] + [u for u in X if u != v]), data=d).fit(disp=0).prsquared), 4)}
           for v in X}
    lo, hi = np.exp(m.conf_int().loc['nop'])
    J['full_model'] = {'n': len(d), 'events': int(d.hard.sum()), 'nop_CI': [round(float(lo), 2), round(float(hi), 2)], 'vars': res}
    # 逐步加入五類控制變數:社經背景與生活成本 → 家庭規模 → 其他收入 → 子女給的錢 → 支出需求(健康)
    steps = [('只控制年齡性別年份', []), ('+ 社經背景、生活成本', ['primary', 'college', 'six']), ('+ 家庭規模', ['co_spouse', 'alone']),
             ('+ 其他收入', ['src_work', 'src_saving', 'n_kid', 'any_full']), ('+ 子女給的錢', ['kid_k']), ('+ 支出需求(完整模型)', ['adl_basic', 'health_bad'])]
    cur, J['stepwise'] = ['nop'], {}
    for lab, add in steps:
        cur = cur + add
        ms = smf.logit('hard ~ ' + ' + '.join(['age', 'female', 'C(wave)'] + cur), data=d).fit(disp=0, **clus(d))
        lo_, hi_ = np.exp(ms.conf_int().loc['nop'])
        J['stepwise'][lab] = [round(float(np.exp(ms.params['nop'])), 2), round(float(lo_), 2), round(float(hi_), 2)]
    log('逐步加入控制變數,沒有按月年金的勝算比:', J['stepwise'])
    af = B.dropna(subset=['hard', 'any_full'])
    J['any_full'] = {'有全職子女(%)': round(float(af.any_full.mean()) * 100, 1),
                     **{f'{"沒年金" if g else "有年金"}_{"有" if k else "沒有"}全職子女_付不出(%)': round(float(af[(af.nop == g) & (af.any_full == k)].hard.mean()) * 100, 1)
                        for g in (0, 1) for k in (1, 0)}}
    log('有全職子女:', J['any_full'])
    d['grpB'] = (d.grp3 == 'B 一次領完型').astype(float); d['grpC'] = (d.grp3 == 'CD 制度外型').astype(float)
    m2 = smf.logit('hard ~ grpB + grpC + ' + ' + '.join(['age', 'female', 'C(wave)'] + [u for u in X if u != 'nop']), data=d).fit(disp=0, **clus(d))
    J['full_model']['一次領完_OR'] = round(float(np.exp(m2.params['grpB'])), 2); J['full_model']['從未退休_OR'] = round(float(np.exp(m2.params['grpC'])), 2)
    log(f'拆成兩類(對照月領年金):一次領完 OR={J["full_model"]["一次領完_OR"]}、從未辦過退休 OR={J["full_model"]["從未退休_OR"]}')
    log(f'完整模型 n={len(d)}:沒有按月年金 OR={res["nop"]["OR"]}(95% CI {lo:.2f} ~ {hi:.2f}),p={res["nop"]["p"]};依拿掉後 R2 減少排序:')
    log(pd.DataFrame(res).T.sort_values('R2_drop', ascending=False).to_string())


# ---------- STEP 6 ----------
def cross_section(A, W2, J, log=print):
    """同一時間點比較:生活費、快樂感、行動、寂寞、可依靠家人;圖 1 的兩組數字"""
    def cmp(d, var, name, kind='logit'):
        m, pen, nop, d = adjusted(d, var, kind)
        log(f'  {name}: 月領年金 {pen:.2f} vs 沒有月領年金 {nop:.2f}  p={m.pvalues["nop"]:.4f}  (n={int((d.nop==0).sum())}/{int((d.nop==1).sum())})')
    log('生活費、快樂感、行動、寂寞(控制年齡、性別、年份):')
    cmp(A[A.wave >= 2022], 'hard', '應付生活費有困難(%)')
    cmp(A, 'happy', '快樂感(1-7 分)', 'ols')
    cmp(A, 'adl_basic', '基本活動受限(%)')
    cmp(A[A.wave >= 2022], 'lonely', '經常寂寞(%)')
    cmp(A[A.wave <= 2020], 'sick_rely', '生病時可依靠家人(%)')
    # 圖 1 左:依子女給的錢分組
    x = A[A.wave >= 2022].dropna(subset=['hard', 'kid_total']).copy()
    x['kq'] = pd.cut(x.kid_total, [-1, 2999, 7999, 1e9], labels=['0–2,999元', '3,000–7,999元', '8,000元以上'])
    log('圖 1 左:生活費有困難(%) × 子女給的錢:\n' + (x.pivot_table(index='kq', columns='nop', values='hard', aggfunc='mean', observed=True) * 100).round(1).to_string())
    tk = x.pivot_table(index='kq', columns='nop', values='hard', aggfunc=['mean', 'size'], observed=True)
    J['hard_by_kid'] = {str(q): {str(int(g)): [round(float(tk['mean'].loc[q, g]) * 100, 1), int(tk['size'].loc[q, g])] for g in [0, 1]} for q in tk.index}
    # 圖 1 右:沒有年金者依最主要經濟來源
    hs = W2[W2.nop == 1].dropna(subset=['hard', 'main']).groupby('main').hard.agg(['mean', 'size'])
    hs = hs[hs['size'] >= 20]
    J['hard_by_main'] = {k: [round(float(v['mean']) * 100, 1), int(v['size'])] for k, v in hs.iterrows()}
    log('圖 1 右:沒有年金者依最主要經濟來源,生活費有困難(%, 人次):', J['hard_by_main'])


# ---------- STEP 7 ----------
def body_function(A, Z, J, log=print):
    """10 項身體功能(Holm 多重比較校正)與 13 種慢性病"""
    adl_p = {}
    for nm in ADL_NAME:
        dd = A.dropna(subset=[nm])
        m = smf.logit(f'Q("{nm}") ~ nop + age + female + C(wave)', data=dd).fit(disp=0, **clus(dd))
        adl_p[nm] = m.pvalues['nop']
        pen, nop = m.predict(dd.assign(nop=0)).mean() * 100, m.predict(dd.assign(nop=1)).mean() * 100
        J.setdefault('adl_items', {})[nm] = {'pen': round(float(pen), 1), 'nop': round(float(nop), 1), 'p': float(m.pvalues['nop'])}
        log(f'  {nm}: 月領年金 {pen:.1f}% / 沒有 {nop:.1f}%  p={m.pvalues["nop"]:.4f}')
    J['adl_n'] = {'pen': int((A.nop == 0).sum()), 'nop': int((A.nop == 1).sum())}
    log(f'  Holm 校正後仍顯著:{multipletests(list(adl_p.values()), method="holm")[0].sum()} / {len(adl_p)} 項')
    dis_p = {}
    for nm in DIS.values():
        dd = Z.dropna(subset=[nm])
        if dd[nm].sum() < 15:
            log(f'  {nm}: 病例 {int(dd[nm].sum())} 人,少於 15 人不比較'); continue
        m = smf.logit(f'Q("{nm}") ~ nop + age + female', data=dd).fit(disp=0); dis_p[nm] = m.pvalues['nop']
    log(f'  慢性病 {len(dis_p)} 種:原始 p<0.05 {sum(p < 0.05 for p in dis_p.values())} 種(n={int((Z.nop==0).sum())}/{int((Z.nop==1).sum())})')


# ---------- STEP 8 ----------
def shocks(T, J, log=print):
    """兩波配對:遇到衝擊時快樂感的變化;「衝擊 × 沒有年金」交互作用"""
    d = T.dropna(subset=['d_happy', 'happy0'])
    m = smf.ols('d_happy ~ any_shock * nop + happy0 + age0 + female + C(y0)', data=d).fit(**clus(d))
    for g, nm in [(0, '月領年金'), (1, '沒有月領年金')]:
        s = d[d.nop == g]
        ms = smf.ols('d_happy ~ any_shock + happy0 + age0 + female + C(y0)', data=s).fit(**clus(s))
        lo, hi = ms.conf_int().loc['any_shock']
        J.setdefault('shock', {})['pen' if g == 0 else 'nop'] = [round(float(ms.params['any_shock']), 3), round(float(lo), 3), round(float(hi), 3), int(s.any_shock.sum()), len(s)]
        log(f'  {nm}:遇到衝擊快樂感變化 {ms.params["any_shock"]:+.3f}(95% CI {lo:.3f} ~ {hi:.3f});遇到衝擊 {int(s.any_shock.sum())} / 配對 {len(s)}')
    J['shock']['inter'] = [round(float(m.params['any_shock:nop']), 3), float(m.pvalues['any_shock:nop'])]
    log(f'  差距(交互作用 any_shock:nop){m.params["any_shock:nop"]:+.3f},p={m.pvalues["any_shock:nop"]:.3f};{len(d)} 組配對、{d.id.nunique()} 人')
    return d


# ---------- STEP 9 ----------
def sandwich_profile(A, log=print):
    """困難長輩(沒有按月年金 + 付不出生活費)的輪廓,以及其中有多少人領到補助"""
    B = A[A.wave >= 2022]
    g = B[B.nop == 1].dropna(subset=['hard', 'lowinc_sub'])
    h = g[g.hard == 1]
    log(f'困難長輩 {len(h)} 人次:領到補助 {h.lowinc_sub.mean()*100:.1f}%({int(h.lowinc_sub.sum())} 人次);平均 {h.age.mean():.1f} 歲、女性 {h.female.mean()*100:.1f}%、'
        f'行動受限 {h.adl_basic.mean()*100:.1f}%、至少一名子女全職 {h.any_full.mean()*100:.1f}%、子女合計給錢中位數 {h.kid_total.median():.0f} 元')
    log('困難長輩的年金類型:', h.grp3.value_counts().to_dict())


# ---------- STEP 10 ----------
def gap_table(D, log=print):
    """表 1:困難長輩依本人領取的津貼,每人每月可用金額與缺口(領補助者的收入未含補助,缺口偏高)"""
    S = D[(D.need == 1) & D.gap.notna()]
    t1 = S.groupby('type').agg(n=('gap', 'size'), 每人可用中位數=('pc_res', 'median'), 平均缺口=('gap', 'mean'))
    t1.loc['合計'] = [len(S), S.pc_res.median(), S.gap.mean()]
    log('表 1:' + chr(10) + t1.round(0).to_string())
    low = S[S.kid_total.notna()]
    log(f'子女給的錢不到最低生活費:{(low.kid_total < low.th).mean()*100:.1f}%(n={len(low)})')
    Sh = S.dropna(subset=['hh_inc_m', 'hh_n'])
    log(f'敏感度-全家總收入÷人數:{len(Sh)} 人次({len(Sh)/len(S)*100:.0f}%),平均缺口 {(Sh.th - Sh.hh_inc_m / Sh.hh_n).clip(lower=0).mean():.0f} 元')
    return t1


# ---------- STEP 12 ----------
def diagnostics(A, d, Wf, log=print):
    """前提檢定與穩健性檢查。d 為 shocks() 用的配對資料"""
    d1 = A[A.wave >= 2022].dropna(subset=['hard', 'age', 'female']).copy()
    f1 = 'hard ~ nop + age + female + C(wave)'
    m1 = smf.logit(f1, data=d1).fit(disp=0, **clus(d1)); m1n = smf.logit(f1, data=d1).fit(disp=0)
    log(f'EPV = {int(d1.hard.sum())} 個事件 ÷ {len(m1.params)-1} 個自變項 = {d1.hard.sum()/(len(m1.params)-1):.0f}(建議 ≥ 10)')
    X = sm.add_constant(pd.get_dummies(d1[['nop', 'age', 'female', 'wave']].astype({'wave': str}), drop_first=True).astype(float))
    log('VIF:', {c: round(variance_inflation_factor(X.values, i), 2) for i, c in enumerate(X.columns) if c != 'const'})
    lr = 2 * (smf.logit('hard ~ nop + age + I(age**2) + female + C(wave)', data=d1).fit(disp=0).llf - m1n.llf)
    log(f'年齡平方項概似比檢定 p = {stats.chi2.sf(lr, 1):.3f}')
    d1['p'] = m1n.predict(d1); d1['dec'] = pd.qcut(d1.p, 10, labels=False, duplicates='drop')
    gq = d1.groupby('dec').agg(o=('hard', 'sum'), e=('p', 'sum'), n=('hard', 'size'))
    hl = (((gq.o - gq.e) ** 2) / (gq.e * (1 - gq.e / gq.n))).sum()
    log(f'Hosmer-Lemeshow p = {stats.chi2.sf(hl, len(gq) - 2):.3f};AUC = {roc_auc_score(d1.hard, d1.p):.3f};OR = {np.exp(m1.params["nop"]):.2f}'
        f'(95% CI {np.exp(m1.conf_int().loc["nop"][0]):.2f} ~ {np.exp(m1.conf_int().loc["nop"][1]):.2f})')
    dd = d1.dropna(subset=['edu_g', 'alone'])
    mm = smf.logit(f1 + ' + C(edu_g) + alone', data=dd).fit(disp=0, **clus(dd))
    log(f'加控教育、獨居:OR = {np.exp(mm.params["nop"]):.2f},p = {mm.pvalues["nop"]:.1e}')
    d2 = A.dropna(subset=['happy', 'age', 'female'])
    m2 = smf.ols('happy ~ nop + age + female + C(wave)', data=d2).fit()
    log(f'快樂感 OLS:Breusch-Pagan p = {het_breuschpagan(m2.resid, m2.model.exog)[1]:.3f}(<0.05 → 需用穩健標準誤,已使用)')
    om = OrderedModel(d2.happy.astype(int), pd.get_dummies(d2[['nop', 'age', 'female', 'wave']].astype({'wave': str}), drop_first=True).astype(float),
                      distr='logit').fit(method='bfgs', disp=0)
    log(f'順序邏輯斯迴歸:nop 係數 {om.params["nop"]:.3f},p = {om.pvalues["nop"]:.4f}')
    for form, nm in [('d_happy ~ any_shock * nop + age0 + female + C(y0)', '不控前一波快樂感'),
                     ('d_happy ~ any_shock * nop + n_shock + happy0 + age0 + female + C(y0)', '加控衝擊件數')]:
        dd = d.assign(n_shock=d[SHOCKS].fillna(0).sum(axis=1))
        mr = smf.ols(form, data=dd).fit(**clus(dd))
        log(f'衝擊交互作用-{nm}:{mr.params["any_shock:nop"]:+.3f},p = {mr.pvalues["any_shock:nop"]:.3f}')
    log(f'遇到衝擊比例:有年金 {d[d.nop==0].any_shock.mean()*100:.1f}% / 沒有年金 {d[d.nop==1].any_shock.mean()*100:.1f}%')
    rows = []
    for y0 in [2018, 2020, 2022]:
        a = Wf[y0]; a = a[(a.age >= 60) & a.grp3.notna()]
        rows.append(pd.DataFrame({'id': a.index, 'y0': y0, 'stay': a.index.isin(Wf[y0 + 2].index).astype(float),
                                  'nop': (a.grp3 != 'A 月領年金').astype(float).values, 'happy0': a.happy.values,
                                  'age0': a.age.values, 'female': 1 - a.male.values, 'adl0': a.adl_basic.values}))
    Rr = pd.concat(rows).dropna()
    ma = smf.logit('stay ~ nop + happy0 + adl0 + age0 + female + C(y0)', data=Rr).fit(disp=0, **clus(Rr))
    log(f'追蹤流失:下一波仍受訪 有年金 {Rr[Rr.nop==0].stay.mean()*100:.1f}% / 沒有年金 {Rr[Rr.nop==1].stay.mean()*100:.1f}%;'
        f'流失模型 OR:沒年金 {np.exp(ma.params["nop"]):.2f}、行動受限 {np.exp(ma.params["adl0"]):.2f}、前一波快樂感 p = {ma.pvalues["happy0"]:.2f}')
