# -*- coding: utf-8 -*-
"""一鍵重現報告所有數字與圖表(STEP 1–12)
    python run_all.py              # 拔靴法 2,000 次,約需數分鐘
    python run_all.py --boot 200   # 測試用:少跑幾次
輸出:results/analysis_pipeline_result.txt(全流程文字輸出)、results/pipeline_result.json(畫圖用數字)、figures/*.png"""
import argparse
import json
import warnings

from pension import analyses as An, estimate as E, figures as F
from pension.config import RESULTS_DIR, N_BOOT
from pension.data import load_raw, build_long, pension_groups, add_children, children_all, main_source, disease_2024, panel_pairs, income

warnings.filterwarnings('ignore')
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def main(n_boot=N_BOOT):
    J = {}
    P('=== STEP 1 ==='); RAW = load_raw(P)                       # 讀入五波
    P('=== STEP 2 ===')                                           # 各波轉成分析變數(在 build_long 內)
    P('=== STEP 3 ==='); L = build_long(RAW, P)                  # 長格式
    P('=== STEP 4 ==='); M, A = pension_groups(RAW, L, P)        # 年金分組
    P('=== STEP 5 ==='); A, K = add_children(RAW, A, P)          # 子女
    P('=== STEP 5b ==='); An.child_trend(*children_all(RAW, L), J, P)        # 檢驗「有子女就有人養」
    P('=== STEP 6 ==='); An.cross_section(A, main_source(RAW, A), J, P)
    P('=== STEP 6b ==='); An.full_model(A, K, J, P)                            # 控制更多變數
    P('=== STEP 7 ==='); An.body_function(A, disease_2024(RAW, A), J, P)
    P('=== STEP 8 ==='); T, Wf = panel_pairs(M); d = An.shocks(T, J, P)
    P('=== STEP 9 ==='); An.sandwich_profile(A, P)
    P('=== STEP 10 ===')
    D = E.add_gap(E.sandwich_frame(A, income(RAW)))
    An.gap_table(D, P)
    E.funnel(D, J, P)
    E.income_profile(D, J, P)
    P('=== STEP 11 ===')
    est = E.stats_of(D, E.weights(D))
    g6 = D.groupby('six').need.mean()
    P(f'樣本 {len(D)} 人次、{D.id.nunique()} 人;困難長輩比例 六都 {g6[1]*100:.1f}% / 非六都 {g6[0]*100:.1f}%')
    Bt = E.bootstrap(D, n_boot)
    J['national'] = {}
    for k, (lo, hi) in E.ci(Bt).items():
        P(f'  {k}: {est[k]:.1f}(95% CI {lo:.1f} ~ {hi:.1f})')
        J['national'][k] = [round(float(est[k]), 1), round(lo, 1), round(hi, 1)]
    psu_rows, strata = E.design(D)
    J['bootstrap'] = {'n_boot': n_boot, '人次': len(D), '人數': int(D.id.nunique()), '分層(縣市)': len(strata), 'PSU(鄉鎮市區)': len(psu_rows),
                      '只有一個鄉鎮的分層': [k for k, v in strata.items() if len(v) == 1],
                      '比例與缺口的相關': round(float(Bt['方法A人數(萬)'].corr(Bt['平均每月缺口(元)'])), 3)}
    P(f"  拔靴設計:{J['bootstrap']['分層(縣市)']} 個縣市、{J['bootstrap']['PSU(鄉鎮市區)']} 個鄉鎮市區;重抽中人數與缺口的相關 {J['bootstrap']['比例與缺口的相關']}")
    P('=== STEP 11b ==='); Bh, est_h = E.hidden_summary(D, J, n_boot, P)    # 被隱藏的中低收入戶與兩階段成本
    P('=== STEP 12 ==='); An.diagnostics(A, d, Wf, P)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / 'analysis_pipeline_result.txt').write_text('\n'.join(OUT), encoding='utf-8')
    (RESULTS_DIR / 'pipeline_result.json').write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding='utf-8')
    for f in [F.fig_child_trend(J), F.fig_missing_piece(J), F.fig_formula(), F.fig_hidden_funnel(J), F.fig_bootstrap(Bh, est_h)]:
        print('圖:', f.name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=N_BOOT, help='拔靴次數')
    main(ap.parse_args().boot)
