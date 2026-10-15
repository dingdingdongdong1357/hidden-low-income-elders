# -*- coding: utf-8 -*-
"""報告與說明文件的圖。輸入為 run_all.py 產生的 J(results/pipeline_result.json),只畫彙總數字"""
import numpy as np
import matplotlib
import matplotlib.patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .config import FIG_DIR, FONTS, PEN, NOP, INK, INK2, MUTED, GRID, SURF

_names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.family'] = next((f for f in FONTS if f in _names), 'sans-serif')
plt.rcParams['axes.unicode_minus'] = False


def style(ax, grid='y'):
    ax.set_facecolor(SURF)
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c6bf'); ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig(w, h):
    plt.rcParams['axes.unicode_minus'] = False                     # notebook 的 inline 後端可能重設,畫圖前再設一次
    return plt.figure(figsize=(w, h), dpi=200, facecolor=SURF)


def title(f, t, s):
    f.text(0.02, 0.97, t, fontsize=14, fontweight='bold', color=INK, va='top')
    f.text(0.02, 0.905, s, fontsize=9.5, color=MUTED, va='top')


def save(f, n):
    FIG_DIR.mkdir(exist_ok=True)
    f.savefig(FIG_DIR / n, facecolor=SURF, bbox_inches='tight', pad_inches=0.25); plt.close(f)
    return FIG_DIR / n


def fig_bootstrap(Bt, est):
    """拔靴法:左 = 2,000 次重抽的(人數, 缺口)散布與等總額曲線;右 = 總金額的分布與兩種區間"""
    f = fig(9.4, 4.0)
    title(f, '人數和每人金額，同一次重抽一起算再相乘', f'{len(Bt):,} 次拔靴重抽；左圖每點是一次重抽，右圖為相乘後的第一階段每年成本')
    ax = f.add_axes([0.07, 0.13, 0.42, 0.62]); style(ax, 'both')
    x, y, t = Bt['隱藏中低收入(萬)'], Bt['每人每月增加(元)'], Bt['第一階段(億)']
    lo, hi = t.quantile([0.025, 0.975])
    ax.scatter(x, y, s=4, color=NOP, alpha=0.25, lw=0)
    x0, x1 = x.min() * 0.97, x.max() * 1.03; y0, y1 = y.min() * 0.95, y.max() * 1.05
    xs = np.linspace(x0, x1, 200)
    for tot, ls in [(lo, '--'), (est['第一階段(億)'], '-'), (hi, '--')]:
        ax.plot(xs, tot * 1e8 / (xs * 1e4 * 12), color=INK2, lw=1, ls=ls)
        c = tot * 1e8 / (1e4 * 12)                                      # 等總額曲線:每人金額 = c ÷ 人數
        if c / x1 >= y0:                                               # 從右邊界離開 → 標在右邊
            ax.text(x1, c / x1 * 1.03, f'{tot:.0f} 億', fontsize=8.5, color=INK2, ha='right', va='bottom', bbox=dict(facecolor=SURF, edgecolor='none', pad=1))
        else:                                                          # 從下邊界離開 → 標在下方
            ax.text(c / y0 * 1.01, y0 * 1.01, f'{tot:.0f} 億', fontsize=8.5, color=INK2, ha='left', va='bottom', bbox=dict(facecolor=SURF, edgecolor='none', pad=1))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel('被隱藏的中低收入戶（萬人）', fontsize=9.5, color=MUTED); ax.set_ylabel('每人每月增加（元）', fontsize=9.5, color=MUTED)
    ax2 = f.add_axes([0.6, 0.13, 0.37, 0.62]); style(ax2, 'y')
    ax2.hist(t, bins=40, color=NOP, alpha=0.8)
    ax2.axvline(est['第一階段(億)'], color=INK, lw=1.2)
    for v in (lo, hi): ax2.axvline(v, color=INK2, lw=1, ls='--')
    ax2.set_xlabel('第一階段每年成本（億元）', fontsize=9.5, color=MUTED)
    ax2.set_title(f'95% 信賴區間 {lo:.0f}–{hi:.0f} 億（中間 95% 的重抽結果）', fontsize=9.5, color=INK2, loc='left')
    return save(f, 'fig_bootstrap.png')


def fig_child_trend(J):
    """圖 1:每年有多少長輩從子女拿到的錢,不到居住縣市的最低生活費(100% 堆疊長條);下方標出中位數與最低生活費"""
    T = J['child_trend']; ys = sorted(T, key=int)
    ok = [T[y]['達最低生活費(%)'] for y in ys]; med = [T[y]['每位長輩合計中位數'] for y in ys]
    line = [T[y]['最低生活費(臺灣)'] for y in ys]; n = [T[y]['長輩數'] for y in ys]
    f = fig(9.4, 3.9)
    title(f, '每年約八成長輩，子女給的錢不夠用',
          '是否達到居住縣市最低生活費（%）；2014 年（斜線）問法不同，僅供參考' + chr(10) + '＊臺灣（直轄市以外）標準；直轄市依各自標準')
    ax = f.add_axes([0.05, 0.27, 0.93, 0.46]); ax.axis('off'); ax.set_xlim(-0.6, len(ys) - 0.4); ax.set_ylim(0, 100)
    for k, (y, o) in enumerate(zip(ys, ok)):
        lo = 100 - o
        b1 = ax.bar(k, lo, 0.62, color=NOP, zorder=2)
        b2 = ax.bar(k, o, 0.62, bottom=lo, color='#d9dee0', zorder=2)
        if T[y]['每位子女中位數'] is None:
            for b in (b1[0], b2[0]):
                b.set_hatch('///'); b.set_edgecolor(SURF); b.set_linewidth(0)
        ax.text(k, lo / 2, f'{lo:.0f}%', ha='center', va='center', fontsize=12, color='white', fontweight='bold',
                bbox=dict(facecolor=NOP, edgecolor='none', pad=2))
        ax.text(k, lo + o / 2, f'{o:.0f}%', ha='center', va='center', fontsize=10, color=INK2)
        ax.text(k, -9, str(y), ha='center', va='top', fontsize=11, color=INK, fontweight='bold')
        ax.text(k, -26, f'子女給的中位數 {med[k]:,.0f} 元', ha='center', va='top', fontsize=8.6, color=INK2)
        ax.text(k, -40, f'最低生活費 {line[k]:,.0f} 元＊', ha='center', va='top', fontsize=8.6, color=MUTED)
        ax.text(k, -54, f'n={n[k]}', ha='center', va='top', fontsize=8, color=MUTED)
    ax.text(-0.6, 106, '■', color=NOP, fontsize=11, va='bottom'); ax.text(-0.42, 106, '不到最低生活費', fontsize=9.5, color=INK2, va='bottom')
    ax.text(1.0, 106, '■', color='#d9dee0', fontsize=11, va='bottom'); ax.text(1.18, 106, '達到最低生活費', fontsize=9.5, color=INK2, va='bottom')
    return save(f, 'fig_child_trend.png')


def fig_missing_piece(J):
    """圖 2:左 = 子女錢不夠的長輩中,過得去 vs 付不出的其他收入;右 = 付不出生活費的邏輯斯迴歸勝算比(95% 信賴區間)"""
    pr = J['income_profile']['profile']; fm = J['full_model']['vars']
    f = fig(9.4, 4.3)
    title(f, '按月年金，是付得出生活費的最關鍵因素',
          '左：子女給不夠的長輩，有各項收入的比例；右：付不出生活費的勝算比（控制其他因素）')
    # 左:啞鈴圖
    ax = f.add_axes([0.17, 0.1, 0.3, 0.64]); style(ax, 'x')
    names = ['按月年金', '儲蓄、利息、租金', '自己或配偶工作', '政府津貼(國保、老農等)']
    ok, bad = pr['過得去'], pr['付不出']
    for i, nm in enumerate(names[::-1]):
        a, b = ok[nm], bad[nm]
        ax.plot([a, b], [i, i], color='#c3c6bf', lw=2.4, zorder=1)
        ax.scatter(a, i, s=60, color=PEN, zorder=3, edgecolor=SURF, lw=1.2)
        ax.scatter(b, i, s=60, color=NOP, zorder=3, edgecolor=SURF, lw=1.2)
        lo_, hi_ = (a, b) if a < b else (b, a)
        ax.text(lo_ - 2.5, i, f'{lo_:.0f}%', va='center', ha='right', fontsize=8.3, color=PEN if lo_ == a else NOP)
        ax.text(hi_ + 2.5, i, f'{hi_:.0f}%', va='center', ha='left', fontsize=8.3, color=PEN if hi_ == a else NOP)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace('(', '（').replace(')', '）') for n in names[::-1]], fontsize=9, color=INK2); ax.set_xlim(0, 90)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{int(v)}%'))
    ax.scatter([], [], color=PEN, s=45, label=f"過得去（n={ok['n']}）"); ax.scatter([], [], color=NOP, s=45, label=f"付不出（n={bad['n']}）")
    ax.legend(frameon=False, fontsize=8.5, loc='lower left', bbox_to_anchor=(-0.05, 1.0), ncol=2, labelcolor=INK2)
    # 右:勝算比森林圖,依五類分組
    groups = [('主要因素', [('nop', '沒有按月年金')]),
              ('其他收入', [('src_saving', '有存款、利息、租金收入'), ('src_work', '自己或配偶有工作收入'), ('kid_k', '子女每月給的錢（每 1,000 元）'),
                         ('any_full', '有全職工作的子女'), ('n_kid', '子女數（每多 1 位）')]),
              ('支出需求', [('health_bad', '自評健康差'), ('adl_basic', '行動受限')]),
              ('家庭規模', [('co_spouse', '與配偶同住'), ('alone', '獨居')]),
              ('社經背景與生活成本', [('primary', '教育：國小以下'), ('college', '教育：大學以上'), ('six', '住六都')])]
    rows = []
    for g, items in groups:
        rows.append(('header', g))
        rows += [('var', v, nm) for v, nm in items]
    ax2 = f.add_axes([0.72, 0.06, 0.25, 0.76]); style(ax2, 'x'); ax2.set_xscale('log')
    ys, labels = [], []
    for k, r in enumerate(rows):
        y = len(rows) - 1 - k
        if r[0] == 'header':
            ax2.text(-0.03, y, r[1], fontsize=8.4, color=INK, fontweight='bold', va='center', ha='right', transform=ax2.get_yaxis_transform())
            continue
        v = r[1]; o, (lo, hi) = fm[v]['OR'], fm[v]['CI']
        col = NOP if v == 'nop' else (INK2 if fm[v]['p'] < 0.05 else MUTED)
        ax2.plot([lo, hi], [y, y], color=col, lw=1.6 if v == 'nop' else 1.1)
        ax2.scatter(o, y, s=55 if v == 'nop' else 26, color=col, zorder=3, marker='D' if v == 'nop' else 'o')
        ax2.text(7.5, y, f'{o:.2f}', va='center', fontsize=7.8, color=col, fontweight='bold' if v == 'nop' else 'normal')
        ys.append(y); labels.append((r[2], v))
    ax2.axvline(1, color=INK, lw=0.9)
    ax2.set_yticks(ys); ax2.set_yticklabels([l for l, _ in labels], fontsize=7.9, color=INK2)
    for t, (_, v) in zip(ax2.get_yticklabels(), labels):
        if v == 'nop': t.set_color(NOP); t.set_fontweight('bold')
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xlim(0.15, 7); ax2.set_xticks([0.25, 0.5, 1, 2, 4]); ax2.set_xticklabels(['0.25', '0.5', '1', '2', '4'], fontsize=8.5); ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel('勝算比（大於 1 較易付不出；灰色為不顯著）', fontsize=8.3, color=MUTED)
    return save(f, 'fig_missing_piece.png')


def fig_hidden_funnel(J):
    """圖 3:被隱藏的中低收入戶,由寬到嚴一層一層篩;每層標出人數與兩階段每年(每月)成本"""
    h = J['hidden']; c = h['cost']; n0 = h['desc']['沒年金長輩(萬)']
    levels = [(n0, '沒有按月年金的長輩', '經濟來源沒有自己或配偶的退休金', None),
              (c['主要結果']['隱藏中低收入(萬)'][0], '① 寬鬆：被隱藏的中低收入戶', '收入（本人與配偶＋子女實際給的錢）' + chr(10) + '符合津貼標準，但沒有領補助', '主要結果'),
              (c['排除有存款利息收入者']['隱藏中低收入(萬)'][0], '② 保守：再排除有存款利息收入者', '經濟來源勾選「儲蓄、利息、租金或投資」者' + chr(10) + '（他們付不出生活費的比例明顯較低）', '排除有存款利息收入者')]
    f = fig(9.4, 3.6)
    title(f, '被隱藏的中低收入戶：146–185 萬人', '第 ② 層是第 ① 層的一部分；右側為兩階段成本，每年（括號為每月）')
    ax = f.add_axes([0.02, 0.03, 0.96, 0.8]); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, len(levels))
    cols = ['#d9dee0', NOP, '#f08a55']
    ax.text(62, len(levels) - 0.02, '第一階段：改領津貼（差額）', fontsize=9, color=MUTED, va='top')
    ax.text(82, len(levels) - 0.02, '第二階段：補足到最低生活費', fontsize=9, color=MUTED, va='top')
    for i, (n, lab, sub, key) in enumerate(levels):
        y = len(levels) - i - 0.62
        ax.text(0, y + 0.12, lab, fontsize=9.6, color=INK, va='center', fontweight='bold')
        ax.text(0, y - 0.2, sub, fontsize=8, color=MUTED, va='center', linespacing=1.2)
        w = n / n0 * 18
        ax.add_patch(matplotlib.patches.Rectangle((33, y - 0.25), w, 0.5, color=cols[i], lw=0))
        ax.text(33 + w + 0.8, y, f'{n:.0f} 萬人', fontsize=11, fontweight='bold', color=INK, va='center')
        if key:
            s1, s2 = c[key]['第一階段(億)'][0], c[key]['第二階段(億)'][0]
            ax.text(62, y + 0.1, f'{s1:,.0f} 億元/年', fontsize=11, fontweight='bold', color=INK, va='center')
            ax.text(62, y - 0.2, f'（約 {s1/12:,.0f} 億元/月）', fontsize=8.5, color=INK2, va='center')
            ax.text(82, y + 0.1, f'再加 {s2:,.0f} 億元/年', fontsize=11, fontweight='bold', color=INK2, va='center')
            ax.text(82, y - 0.2, f'（約 {s2/12:,.0f} 億元/月）', fontsize=8.5, color=INK2, va='center')
    return save(f, 'fig_hidden_funnel.png')


def fig_formula():
    """2.2 節的試算公式(LaTeX 語法,matplotlib mathtext 排版)"""
    plt.rcParams['mathtext.fontset'] = 'cm'
    f = plt.figure(figsize=(9.4, 3.0), dpi=220, facecolor='white')
    ax = f.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    rows = [('（1）每人每月可用的錢', r'$y_i=\dfrac{W_i+B_i+C_i}{n_i}$'),
            ('（2）應領津貼', None),
            ('（3）擇一後實際增加的錢', r'$g_i=\max\left(A_i-E_i,\ 0\right)$'),
            ('（4）被隱藏的中低收入戶', r'$H_i=1$，若 $A_i>0$ 且 $S_i=0$'),
            ('（5）第一階段成本（每年）', r'$C_1=12\,N\cdot\mathrm{E}_w\left[H_i\,g_i\right]$'),
            ('（6）第二階段成本（每年）', r'$C_2=12\,N\cdot\mathrm{E}_w\left[\max\left(L_i-y_i-g_i,\ 0\right)\right]$')]
    ys = [91, 68, 46, 33, 20, 7]
    for (lab, fx), y in zip(rows, ys):
        ax.text(1, y, lab, fontsize=11, color='#1b1f24', va='center')
        if fx:
            ax.text(30, y, fx, fontsize=14.5, color='#1b1f24', va='center')
    # (2) 分段函數:手動畫大括號
    ax.text(30, 68, r'$A_i=$', fontsize=14.5, va='center')
    ax.text(36.2, 68, '{', fontsize=40, va='center', color='#1b1f24', fontweight='light')
    for dy, txt in [(8, r'$8{,}329$ 元，若 $y_i<1.5\,L_i$'), (0, r'$4{,}164$ 元，若 $1.5\,L_i\leq y_i<2.5\,L_i$'), (-8, r'$0$，其他')]:
        ax.text(39.5, 68 + dy, txt, fontsize=13, va='center', color='#1b1f24')
    f.savefig(FIG_DIR / 'fig_formula.png', facecolor='white', bbox_inches='tight', pad_inches=0.08); plt.close(f)
    return FIG_DIR / 'fig_formula.png'
