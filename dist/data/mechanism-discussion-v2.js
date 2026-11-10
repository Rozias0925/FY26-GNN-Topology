/* Interpretation of the verified complete V2 batch, reviewed 2026-10-06.
   Scores remain in mechanism-results-v2.js; paragraphs do not define winners. */
window.MECHANISM_DISCUSSION = {
  stage_a: {
    zh: [
      String.raw`自身訊號情境中，MLP、GCN、GAT 的平均 \(R^2\) 分別約為 \(0.911\)、\(0.912\)、\(0.911\)。MLP 具有競爭力；約 \(0.001\) 的差距不能解讀為圖提供了重要額外資訊。均勻鄰居情境（\(\lambda=0.9\)、\(\kappa=0\)）中，GCN 與 GAT 約為 \(0.902\)、\(0.903\)，MLP 約為 \(0.138\)：此時鄰居特徵才是自身特徵無法充分取得的訊號。`,
      String.raw`選擇性鄰居情境（\(\lambda=0.9\)、\(\kappa=4\)）中，GAT 約為 \(0.897\)，GCN 約為 \(0.859\)，配對平均差約為 \(+0.038\)。平滑訊號的 denoising 情境中，GCN／GAT 約為 \(0.744\)，MLP 約為 \(0.713\)。這些極端情境確認了可研究的機制；仍需配合單軸曲線及五種子區間解讀。`,
      String.raw`有效動態情境中，Dynamic-GCN／GAT 約為 \(0.901\)／\(0.903\)，Static-GCN／GAT 約為 \(0.136\)／\(0.132\)；只有觀測圖雜訊的情境則相反，Static 約為 \(0.902\)／\(0.903\)，Dynamic 約為 \(0.399\)／\(0.401\)。更新圖的價值取決於變動是否保留目標相關資訊。`
    ],
    en: [
      String.raw`In the feature-only regime, mean \(R^2\) is approximately \(0.911\) for MLP, \(0.912\) for GCN, and \(0.911\) for GAT. MLP is competitive; a gap near \(0.001\) does not establish substantial incremental graph information. Under uniform neighbors (\(\lambda=0.9\), \(\kappa=0\)), GCN and GAT score about \(0.902\) and \(0.903\), versus \(0.138\) for MLP: neighbor features carry signal that local features cannot fully recover.`,
      String.raw`Under selective neighbors (\(\lambda=0.9\), \(\kappa=4\)), GAT scores about \(0.897\) and GCN \(0.859\), a mean paired gap near \(+0.038\). In the smooth-signal denoising regime, GCN/GAT score about \(0.744\) versus \(0.713\) for MLP. These extreme regimes verify mechanisms to investigate; response curves and five-seed intervals remain necessary.`,
      String.raw`With predictive topology drift, Dynamic-GCN/GAT score about \(0.901\)/\(0.903\) versus \(0.136\)/\(0.132\) for their static counterparts. With observation noise alone, Static scores about \(0.902\)/\(0.903\) versus \(0.399\)/\(0.401\) for Dynamic. Updating a graph helps when its changes preserve target-relevant information.`
    ]
  },
  curves: {
    zh: [
      String.raw`\(\lambda\) 從 \(0\) 增至 \(1\) 時，\(\mathrm{GCN}-\mathrm{MLP}\) 的平均 \(\Delta R^2\) 由約 \(+0.001\) 增至 \(+0.852\)，與 \(S_G\) 增加一致。固定 \(\lambda=0.75\)，\(\kappa\) 從 \(0\) 增至 \(4\) 時，\(\mathrm{GAT}-\mathrm{GCN}\) 由約 \(+0.0005\) 增至 \(+0.0339\)；\(\kappa=0\) 的 \(95\%\) 配對區間跨越零，\(\kappa=4\) 的區間約為 \([0.0299, 0.0373]\)。均勻聚合時架構差很小，選擇性聚合時注意力的相對收益增加。`,
      String.raw`獨立 denoising 的 \(\tau\) 從 \(0\) 增至 \(5\) 時，\(\mathrm{GCN}-\mathrm{MLP}\) 由約 \(-0.0002\) 增至 \(+0.0908\)。但由帶雜訊目標估計的 \(H\) 並非單調：\(\tau=2\) 約為 \(0.178\)，\(\tau=5\) 約為 \(0.154\)。因此本結果支持此生成器中的平滑／去雜訊收益，不能將 \(\tau\) 增加直接等同於可觀測 \(H\) 增加。`,
      String.raw`\(\delta>0\) 的乾淨真實變動均使 Dynamic 優於 Static，但收益不是單調曲線：GCN 的 \(\Delta R^2\) 在 \(\delta=0.05\)、\(0.10\)、\(0.40\) 約為 \(+0.658\)、\(+0.710\)、\(+0.640\)。\(D_{\mathrm{pred}}\) 在這些點持續增加，模型收益卻在後段下降；「診斷值越大，收益必然越大」的嚴格版本並未成立。`,
      String.raw`真實圖固定、只增加觀測雜訊時，\(\eta=0.4\) 的 \(\mathrm{GCN}_{D}-\mathrm{GCN}_{S}\) 約為 \(-0.418\)，\(95\%\) 區間約為 \([-0.426, -0.411]\)，GAT 亦呈負差。\(D_{\mathrm{struct}}\) 衡量改變幅度；\(D_{\mathrm{pred}}\) 可以為負，代表更新圖的簡單 probe 比共識圖更差。結構變動本身不足以支持 Dynamic。`
    ],
    en: [
      String.raw`As \(\lambda\) increases from \(0\) to \(1\), the mean \(\mathrm{GCN}-\mathrm{MLP}\) \(\Delta R^2\) rises from about \(+0.001\) to \(+0.852\), alongside increasing \(S_G\). At \(\lambda=0.75\), increasing \(\kappa\) from \(0\) to \(4\) raises \(\mathrm{GAT}-\mathrm{GCN}\) from about \(+0.0005\) to \(+0.0339\). The paired \(95\%\) interval crosses zero at \(\kappa=0\) and is approximately \([0.0299, 0.0373]\) at \(\kappa=4\). Architecture differences are small under uniform aggregation and grow under selective aggregation.`,
      String.raw`In isolated denoising, increasing \(\tau\) from \(0\) to \(5\) raises \(\mathrm{GCN}-\mathrm{MLP}\) from about \(-0.0002\) to \(+0.0908\). However, \(H\) estimated from noisy targets is not monotone: about \(0.178\) at \(\tau=2\) and \(0.154\) at \(\tau=5\). This supports a smoothing/denoising benefit within this generator, without equating larger \(\tau\) with larger observable \(H\).`,
      String.raw`Clean predictive drift with \(\delta>0\) favors Dynamic over Static, but its benefit is not monotone. GCN \(\Delta R^2\) is approximately \(+0.658\), \(+0.710\), and \(+0.640\) at \(\delta=0.05\), \(0.10\), and \(0.40\). \(D_{\mathrm{pred}}\) keeps increasing across these points while model benefit declines after its peak. A strict claim that larger diagnostics always yield larger gains is unsupported.`,
      String.raw`When the true graph is fixed and only observation noise increases, \(\mathrm{GCN}_{D}-\mathrm{GCN}_{S}\) is about \(-0.418\) at \(\eta=0.4\), with a \(95\%\) interval near \([-0.426, -0.411]\); GAT also has a negative gap. \(D_{\mathrm{struct}}\) measures the amount of change. \(D_{\mathrm{pred}}\) can be negative when the dynamic probe performs worse than the consensus-graph probe. Structural change alone does not justify Dynamic.`
    ]
  },
  interactions: {
    zh: [
      String.raw`\(\lambda\times\kappa\) 網格顯示：\(\lambda=0\) 時，改變 \(\kappa\) 不影響目標生成中的自身訊號；\(\mathrm{GAT}-\mathrm{GCN}\) 僅約 \(-0.0007\)。\(\kappa=4\) 時，\(\lambda=0.25\) 與 \(1\) 的 \(\mathrm{GAT}-\mathrm{GCN}\) 分別約為 \(+0.020\)、\(+0.036\)。鄰居重要性不均勻必須與實際圖訊號一起解讀。`,
      String.raw`\(\delta\times\eta\) 網格中，固定 \(\delta=0.10\)，\(\eta\) 從 \(0\) 增至 \(0.40\)，\(\mathrm{GCN}_{D}-\mathrm{GCN}_{S}\) 由約 \(+0.710\) 降至 \(+0.309\)；\(\delta=0\)、\(\eta=0.40\) 時則為 \(-0.418\)。真實變動與估計雜訊同時存在時，應檢查動態圖仍保留多少增量資訊，而不能只看邊變動比例。`,
      String.raw`Stage C 的 \(\tau\) 是平滑圖特徵後再生成下一步目標。固定 \(\lambda=0.75\)，\(\tau\) 從 \(0\) 增至 \(5\) 時，\(S_X\) 由約 \(0.273\) 增至 \(0.633\)，\(S_G\) 由 \(0.644\) 降至 \(0.281\)，\(\mathrm{GCN}-\mathrm{MLP}\) 由 \(+0.640\) 降至 \(+0.227\)。這與「自身特徵可代理更多鄰居資訊」的解釋一致；它與 Stage B 的獨立 denoising 是不同問題，不能合併成一條 \(\tau\) 結論。`
    ],
    en: [
      String.raw`The \(\lambda\times\kappa\) grid shows that \(\kappa\) does not change the self-generated target signal when \(\lambda=0\); \(\mathrm{GAT}-\mathrm{GCN}\) is only about \(-0.0007\). At \(\kappa=4\), the gap is about \(+0.020\) for \(\lambda=0.25\) and \(+0.036\) for \(\lambda=1\). Unequal relevance must be interpreted alongside actual graph signal.`,
      String.raw`In the \(\delta\times\eta\) grid, fixing \(\delta=0.10\) and increasing \(\eta\) from \(0\) to \(0.40\) reduces \(\mathrm{GCN}_{D}-\mathrm{GCN}_{S}\) from about \(+0.710\) to \(+0.309\). At \(\delta=0\) and \(\eta=0.40\) the gap is \(-0.418\). When true drift and observation noise coexist, assess retained incremental information rather than edge turnover alone.`,
      String.raw`Stage C smooths graph features before generating next-step targets. At \(\lambda=0.75\), increasing \(\tau\) from \(0\) to \(5\) raises \(S_X\) from about \(0.273\) to \(0.633\), reduces \(S_G\) from \(0.644\) to \(0.281\), and reduces \(\mathrm{GCN}-\mathrm{MLP}\) from \(+0.640\) to \(+0.227\). This is consistent with local features becoming better proxies for neighbor information. It is a different task from isolated Stage B denoising and requires a separate interpretation.`
    ]
  },
  multi_axis: {
    zh: [
      String.raw`\(200\) 組條件的探索性 Spearman 關聯中，\(S_G\) 與 \(\mathrm{GCN}_{D}-\mathrm{MLP}\) 的 \(\rho\) 約為 \(0.954\)，與 \(\mathrm{GAT}_{D}-\mathrm{MLP}\) 約為 \(0.938\)。\(D_{\mathrm{pred}}\) 與 \(\mathrm{Dynamic}-\mathrm{Static}\) 的 \(\rho\) 在 GCN／GAT 約為 \(0.890\)／\(0.876\)；\(D_{\mathrm{struct}}\) 與 GCN 的同一差值反而約為 \(-0.292\)。診斷圖支持「有預測價值的變動」比「變動幅度」更能描述收益，但這些是多因素共存下的關聯。`,
      String.raw`\(\hat R\) 與 \(\mathrm{GAT}_{D}-\mathrm{GCN}_{D}\) 的 \(\rho\) 約為 \(0.720\)，與 \(\mathrm{GAT}_{S}-\mathrm{GCN}_{S}\) 卻約為 \(-0.315\)。注意力優勢不能直接由單一 \(\hat R\) 決定；共識拓樸與當下生成拓樸的落差可能改變架構的相對作用，仍需配合圖品質及其他診斷檢查。`,
      String.raw`在 \(1{,}000\) 個條件／種子診斷中，\(\hat R\) 與 \(R_{\mathrm{true}}\) 的排序相關約為 \(0.896\)，平均絕對誤差約為 \(0.111\)。能追蹤排序不代表數值已校準。這 \(200\) 組條件皆有 \(\lambda>0\)，Dynamic 相對 MLP 的平均差皆為正，也不能推論無圖訊號或未涵蓋的生成機制都應選 GNN。只有五個種子，逐條件區間未做多重比較校正；本相圖描述已模擬條件的表現差，並非保證選中最佳模型的規則。`
    ],
    en: [
      String.raw`Across \(200\) conditions, exploratory Spearman \(\rho\) is about \(0.954\) between \(S_G\) and \(\mathrm{GCN}_{D}-\mathrm{MLP}\), and \(0.938\) for \(\mathrm{GAT}_{D}-\mathrm{MLP}\). For \(D_{\mathrm{pred}}\) versus \(\mathrm{Dynamic}-\mathrm{Static}\), \(\rho\) is about \(0.890\) for GCN and \(0.876\) for GAT. \(D_{\mathrm{struct}}\) has \(\rho\) near \(-0.292\) with the GCN gap. Predictive value describes benefit better than turnover magnitude here, but these remain associations under coexisting mechanisms.`,
      String.raw`\(\hat R\) has \(\rho\) near \(0.720\) with \(\mathrm{GAT}_{D}-\mathrm{GCN}_{D}\), but \(-0.315\) with the corresponding static contrast. Attention benefit cannot be determined from \(\hat R\) alone. A mismatch between consensus topology and current target-generating topology may change architecture behavior; graph quality and other diagnostics require joint inspection.`,
      String.raw`Across \(1{,}000\) condition/seed diagnostics, \(\hat R\) and \(R_{\mathrm{true}}\) have rank correlation near \(0.896\) and mean absolute error near \(0.111\). Tracking ranks does not establish numerical calibration. Every sampled condition has \(\lambda>0\); positive mean \(\mathrm{Dynamic}-\mathrm{MLP}\) gaps throughout this sample do not imply that GNNs suit conditions without graph signal or unstudied generators. There are only five seeds, and condition-level intervals have no multiple-comparison adjustment. The map describes gaps in simulated conditions rather than guaranteeing the best model.`
    ]
  },
  sample_size: {
    zh: [
      String.raw`\(N\in\{100,300,500,1000\}\) 時，\(\mathrm{GAT}-\mathrm{GCN}\) 的平均 \(\Delta R^2\) 約為 \(+0.0031\)、\(+0.0011\)、\(+0.0005\)、\(+0.0010\)。前三組 \(95\%\) 區間跨越零；\(N=1000\) 的區間約為 \([0.0001, 0.0019]\)。差異整體很小，這次未取得「小樣本時 GCN 明顯更好」的證據，也沒有進行統計等效性檢定。`,
      String.raw`這組固定 \(\kappa=0\)、\(\lambda=0.75\)、乾淨 regular graph，並保留五個獨立資料種子。Static／Dynamic 使用完全相同的固定拓樸，依既定等價訓練重用規則同分；這些同分不是額外獨立重複。增加節點數亦不等於增加獨立資料集的數量。`
    ],
    en: [
      String.raw`At \(N\in\{100,300,500,1000\}\), mean \(\mathrm{GAT}-\mathrm{GCN}\) \(\Delta R^2\) is approximately \(+0.0031\), \(+0.0011\), \(+0.0005\), and \(+0.0010\). The first three \(95\%\) intervals cross zero; at \(N=1000\) the interval is roughly \([0.0001, 0.0019]\). Differences are small. This run does not establish a clear small-sample GCN advantage, and no statistical equivalence test was conducted.`,
      String.raw`These conditions retain \(\kappa=0\), \(\lambda=0.75\), a clean regular graph, and five independent dataset seeds. Static/Dynamic use identical fixed topology and share scores under the prespecified equivalent-training reuse rule; these are not additional independent replications. More nodes do not mean more independent datasets.`
    ]
  },
  robustness: {
    zh: [
      String.raw`\(\kappa=0\) 時，異質度數圖的 \(\mathrm{GAT}-\mathrm{GCN}\) 約為 \(+0.076\)，SBM 約為 \(+0.020\)，kNN 約為 \(+0.007\)，均大於 regular control 的約 \(+0.0005\)。均勻 target weights 與 GCN 的對稱正規化只有在 regular graph 才完全對齊；離開此假設後，\(\kappa=0\) 不再是相同的 GCN-aligned 情境。`,
      String.raw`反應雜訊 \(\sigma\) 從 \(0.1\) 增至 \(1\)，\(\mathrm{GAT}_{D}-\mathrm{MLP}\) 的 \(\Delta R^2\) 由約 \(+0.700\) 降至 \(+0.333\)。分類另以 \(\mathrm{AP}\) 衡量：\(\kappa=0\) 的 \(\mathrm{GAT}-\mathrm{GCN}\) 約 \(+0.0019\)、區間跨零；\(\kappa=4\) 約 \(+0.0484\)，\(95\%\) 區間約為 \([0.0431, 0.0538]\)。這支持選擇性收益可出現在此合成二元分類生成器；分類 AP 與回歸分數分開解讀。`,
      String.raw`在 query-dependent relevance 情境中，\(\mathrm{GATv2}-\mathrm{GAT}\) 的 \(\Delta R^2\) 約為 \(+0.076\)，\(95\%\) 區間約為 \([0.073, 0.079]\)。在 source-only controls 中則沒有一致的 GATv2 收益。因此額外彈性應對應特定機制，不能把這個結果延伸成 GATv2 普遍更佳。`
    ],
    en: [
      String.raw`At \(\kappa=0\), \(\mathrm{GAT}-\mathrm{GCN}\) is about \(+0.076\) on heterogeneous-degree graphs, \(+0.020\) on SBM, and \(+0.007\) on kNN, compared with about \(+0.0005\) for the regular control. Uniform target weights align exactly with symmetric GCN normalization only on regular graphs. Outside that assumption, \(\kappa=0\) is no longer the same GCN-aligned regime.`,
      String.raw`Increasing response noise \(\sigma\) from \(0.1\) to \(1\) reduces \(\mathrm{GAT}_{D}-\mathrm{MLP}\) \(\Delta R^2\) from about \(+0.700\) to \(+0.333\). Classification uses \(\mathrm{AP}\) separately: \(\mathrm{GAT}-\mathrm{GCN}\) is about \(+0.0019\) with an interval crossing zero at \(\kappa=0\), and \(+0.0484\) with a \(95\%\) interval near \([0.0431, 0.0538]\) at \(\kappa=4\). Selectivity benefit therefore also appears in this synthetic binary generator; classification AP and regression scores are interpreted separately.`,
      String.raw`With query-dependent relevance, \(\mathrm{GATv2}-\mathrm{GAT}\) \(\Delta R^2\) is about \(+0.076\), with a \(95\%\) interval near \([0.073, 0.079]\). Source-only controls do not show a consistent GATv2 benefit. Additional flexibility should correspond to a specific mechanism; this result does not establish universal GATv2 superiority.`
    ]
  },
  convergence: {
    zh: [
      String.raw`將六個 Stage A 情境的訓練上限從 \(200\) 提高至 \(400\) 後，各情境／模型平均 \(R^2\) 變化的最大絕對值約為 \(0.00164\)，出現在有效動態情境的 Dynamic-GCN。自身訊號的近似同分、選擇性情境的 GAT 收益，以及有效／雜訊動態的相反方向均保留。`,
      String.raw`\(400\)-epoch 結果仍有 \(8\)／\(150\) 筆評估記錄標示選中候選達訓練上限；這個檢查也只涵蓋六個機制情境。分數變化小可支持主要對照對預算較不敏感，不能證明所有模型或 Stage C／D 都已收斂。主結果保留原 \(200\)-epoch 預算，不以 test 分數事後挑選較佳預算。`
    ],
    en: [
      String.raw`Raising the training cap from \(200\) to \(400\) for the six Stage A conditions changes each regime/model mean \(R^2\) by at most about \(0.00164\) in absolute value, attained by Dynamic-GCN under predictive drift. Near-equal feature-only scores, selective-neighbor GAT benefit, and opposite predictive/noisy-dynamics directions remain.`,
      String.raw`The \(400\)-epoch results still flag the selected candidate as budget-exhausted in \(8\) of \(150\) evaluation records. This check covers only six mechanism conditions. Small score changes support limited budget sensitivity for these controls, not convergence of every model or of Stages C/D. Primary results retain the \(200\)-epoch budget rather than selecting a budget retrospectively using test scores.`
    ]
  }
};
