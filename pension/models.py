# -*- coding: utf-8 -*-
"""迴歸模型:受訪者群聚標準誤、調整後比例"""
import statsmodels.formula.api as smf


def clus(d):
    """群聚標準誤:同一位受訪者(id)的多筆資料視為一群"""
    return {'cov_type': 'cluster', 'cov_kwds': {'groups': d['id'].astype('int64')}}


def adjusted(d, var, kind='logit'):
    """比較有 / 沒有月領年金,控制年齡、性別(多年資料再加年份)。
    調整後比例 = 把「每一個人」都假設成有年金(nop=0)算一次預測值取平均,再假設成沒有年金(nop=1)算一次,
    兩者的差距只來自年金,不來自兩組年齡、性別組成不同。
    回傳 (模型, 有年金值, 沒有年金值, 使用的資料)"""
    d = d.dropna(subset=[var, 'age', 'female']).copy()
    wv = ' + C(wave)' if d.wave.nunique() > 1 else ''
    f = f'Q("{var}") ~ nop + age + female{wv}'
    m = (smf.ols if kind == 'ols' else smf.logit)(f, data=d)
    m = m.fit(**clus(d)) if kind == 'ols' else m.fit(disp=0, **clus(d))
    k = 1 if kind == 'ols' else 100
    pen, nop = m.predict(d.assign(nop=0)).mean() * k, m.predict(d.assign(nop=1)).mean() * k
    return m, pen, nop, d
