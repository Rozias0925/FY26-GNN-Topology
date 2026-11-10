/* Offline, bilingual views of the completed pilot; no scores are computed by the demo. */
(function () {
  'use strict';
  const data = window.SIMULATION_RESULTS;
  const state = {mode:'observed', horizon:1, scenario:'all', metric:'test_ap'};
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = (x,n=4) => x == null ? '—' : Number(x).toFixed(n);
  const signed = x => x == null ? '—' : `${x > 0 ? '+' : ''}${Number(x).toFixed(4)}`;
  const scenarios = window.SimulationScenarios;
  const modeName = (mode,t) => mode === 'observed' ? t('觀測圖實驗','Observed-graph arm') : t('估計圖實驗','Estimated-graph arm');
  const scenarioLabel = s => `λ=${s.graph_signal} · δ=${s.graph_drift} · η=${s.edge_noise} · κ=${s.neighbor_heterogeneity}`;
  const metricName = key => ({test_ap:'Average Precision (AP)',test_auroc:'AUROC',test_brier:'Brier score',test_rmse:'RMSE'})[key];
  const ci = (row,key) => row[key+'_ci_low'] == null ? '—' : `[${fmt(row[key+'_ci_low'])}, ${fmt(row[key+'_ci_high'])}]`;
  const table = (headers,rows,caption,kind='results-table') => `<div class="table-wrap ${kind}" tabindex="0" role="region" aria-label="${esc(caption)}"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((c,i)=>`<${i?'td':'th scope="row"'}>${c}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const heading = (n,id,zh,en,t) => `<h2 id="${id}" tabindex="-1"><span>${n}</span>${t(zh,en)}</h2>`;

  function intro(t) {
    const sections=[
      ['3.1','sim-task','研究問題與預測目標','Question and targets'],
      ['3.2','sim-generation','資料生成過程','Data generation'],
      ['3.3','sim-scenarios','控制參數與模擬情境','Factors and scenarios'],
      ['3.4','sim-comparison','建圖方法與模型比較','Graphs and models'],
      ['3.5','sim-evaluation','評估指標與選模方法','Evaluation and selection'],
      ['3.6','sim-results','模擬結果','Simulation results'],
      ['3.7','sim-discussion','結果解讀與研究限制','Discussion and limitations']
    ];
    return `<nav class="simulation-outline" aria-label="${t('Simulation 本章目錄','Simulation chapter contents')}"><p class="outline-label">${t('本章閱讀順序','Chapter contents')}</p><ol>${sections.map(([n,id,zh,en])=>`<li><button data-result-jump="${id}"><span>${n}</span>${t(zh,en)}</button></li>`).join('')}</ol></nav>`;
  }

  function runSummary(t) {
    return `<div class="run-summary"><h3>${t('本輪實驗規模','Pilot size')}</h3><p>${t('21 組參數設定 × 3 個 seeds × 3 個預測範圍 × 5 種模型 = 945 組模型評估。每組比較兩個學習率，合計 1,890 次候選訓練。這是 2026-09-30 完成的第一輪因子篩選。','21 settings × 3 seeds × 3 horizons × 5 models = 945 model evaluations. Two learning rates per evaluation give 1,890 candidate fits. This factorial pilot completed on 2026-09-30.')}</p></div>`;
  }

  function methods(t) {
    return `<h3>${t('先定義如何衡量預測表現','Define predictive performance before comparison')}</h3>
      <p>${t('主要比較分類 AP；AUROC、Brier 與回歸 RMSE 提供不同面向的補充。所有分數先在同一情境、seed、h 的測試窗計算，再對三個 seeds 取平均。','Classification AP is primary; AUROC, Brier and regression RMSE provide complementary views. Scores are computed on test windows for each scenario, seed and horizon before averaging over the three seeds.')}</p>
      ${table([t('指標','Metric'),t('定義與讀法','Definition and interpretation'),t('方向','Direction')],[
        ['AP',t('Average Precision：依排序門檻變動，以 recall 的增量加權 precision；衡量正例排序表現。AP 與梯形 PR-AUC 的計算方式不同。','Average Precision weights precision by increments in recall as the ranking threshold changes, measuring positive-event ranking. Its calculation differs from trapezoidal PR-AUC.'),t('越高越好','Higher is better')],
        ['AUROC',t('衡量模型將正例排在負例之前的能力；需與 AP 和測試正例比例一起讀。','Measures the ability to rank positives above negatives; read alongside AP and test prevalence.'),t('越高越好','Higher is better')],
        ['Brier',t('mean[(p̂−y)²]：預測機率 p̂ 與二元標籤 y 的均方誤差。','mean[(p̂−y)²]: mean squared error between probability p̂ and binary label y.'),t('越低越好','Lower is better')],
        ['RMSE',t('√mean[(r̂−r)²]：連續反應的均方根誤差；本輪單位是合成反應尺度。','√mean[(r̂−r)²]: root mean squared error for continuous responses, in generated-response units.'),t('越低越好','Lower is better')]
      ],t('分類為主、回歸為輔的評估指標','Metrics for primary classification and supplementary regression'),'metric-definition-table')}
      <p>${t('以 MLP 作為自身歷史基準，ΔAP = 模型 AP − MLP AP。另報告訓練盛行率與持續值基線。單一情境以配對 seed 差值及 bootstrap 區間呈現不確定性；跨情境平均只作描述性摘要。','MLP supplies the local-history reference: ΔAP = model AP − MLP AP. Training-prevalence and persistence baselines are also reported. Paired seed differences and bootstrap intervals summarize uncertainty within a setting; cross-scenario means are descriptive only.')}</p>`;
  }

  function selectionMethod(t) {
    return `<h3>${t('選模規則如何驗證？','How is the selection rule validated?')}</h3>
      <ol class="numbered-prose"><li><strong>${t('按生成機制留出情境','Hold out generating mechanisms')}</strong><p>${t('將 25% 生成機制群組留作評估。同機制的 seeds、邊雜訊變體及無效參數變體都在同側，避免近似重複的設定同時進入規則訓練與評估。','Hold out 25% of generating-mechanism groups. Seeds, edge-noise variants and ineffective-factor variants stay on the same side, preventing near-duplicate settings from entering both rule training and evaluation.')}</p></li><li><strong>${t('只用開發情境的驗證 AP','Use development validation AP')}</strong><p>${t('用開發情境的指標與驗證 AP 擬合選模樹；最佳固定模型也只依這些情境的驗證 AP 選定。','Fit the tree using descriptors and validation AP from development scenarios. Select the best fixed model using validation AP from those same scenarios.')}</p></li><li><strong>${t('在留出情境比較測試 AP','Compare held-out test AP')}</strong><p>${t('比較規則與固定模型的測試 AP，並報告 regret 與 Gray 比例。結果在 3.6 的 C 部分呈現。','Compare rule and fixed-model test AP, reporting regret and Gray fraction. Results appear in part C of 3.6.')}</p></li></ol>
      <details class="results-details"><summary>${t('選模方法與可用範圍','Selection method and applicability')}</summary><p>${t('輸入為訓練期的 Graph Utility、SDNR、NRC、Dirichlet energy 與 density。規則是深度最多 3、葉節點至少 2 筆的多輸出回歸樹，預測五模型的驗證 AP。按生成機制留出 25% 群組，同機制的 seeds、觀測雜訊變體及無效參數變體都留在同側。','Inputs are training-period Graph Utility, SDNR, NRC, Dirichlet energy and density. A multi-output regression tree, depth at most 3 with at least 2 samples per leaf, predicts the five validation AP scores. A 25% grouped holdout keeps seeds, observation-noise variants and ineffective parameter variants of the same generating mechanism on the same side.')}</p><p>${t('留出評估後才用全部模擬的驗證分數擬合外部使用版本。外部特徵缺失或超出逐欄訓練範圍時拒絕指定單一模型。三個 seeds 使用配對亂數，因此不能把所有情境當成完全獨立的重複。完整樹、適用範圍及留出情境清單可下載檢查。','After holdout evaluation, the deployment rule is refitted using all simulation validation scores. It abstains when external features are missing or outside the per-feature training ranges. Seeds use paired random streams across settings, so scenarios are not wholly independent replicates. The full trees, feature ranges and holdout lists are available below.')}</p></details>
      `;
  }

  function markup(t) {
    return `${heading('3.6','sim-results','模擬結果','Simulation results',t)}
      <p>${t('依序閱讀三個比較：A 看相同生成條件下的圖來源差異；B 查看某個情境的完整指標；C 檢驗選模規則。先選預測步數，A 與 B 會同步更新。','Read three comparisons in order: A compares graph sources under matched conditions; B inspects a setting’s full metrics; C evaluates the selection rule. Choose the horizon first; A and B update together.')}</p>
      <div class="results-controls results-horizon-control"><label for="results-horizon">${t('本節的預測步數','Horizon for this section')}<select id="results-horizon">${data.horizons.map(h=>`<option value="${h}">h = ${h} ${t('步','step(s)')}</option>`).join('')}</select></label><p>${t('A 的 AP 對照表與 B 的詳細結果共用此範圍；C 同時列出三個範圍。','A’s AP comparison and B’s detailed results share this horizon; C lists all three horizons.')}</p></div>
      <section id="results-matched" aria-label="${t('同情境圖來源對照','Matched-condition graph-source comparison')}"></section>
      <h3 id="sim-detail" tabindex="-1">${t('B. 單一情境與完整指標','B. Individual settings and full metrics')}</h3>
      <p>${t('選擇圖來源、具名情境與指標。全部情境選項顯示同一實驗的等權平均；單一情境顯示三個 seeds 的平均與 95% bootstrap 區間。','Select a graph source, named setting and metric. The all-scenarios option shows an equal-weight mean within one arm; an individual setting shows its three-seed mean and 95% bootstrap interval.')}</p>
      <div class="results-controls">
        <label for="results-mode">${t('圖建構方式','Graph construction')}<select id="results-mode"><option value="observed">${t('觀測圖（16 情境）','Observed graph (16 scenarios)')}</option><option value="estimated">${t('估計圖（5 情境）','Estimated graph (5 scenarios)')}</option></select></label>
        <label for="results-metric">${t('圖表指標','Chart metric')}<select id="results-metric"><option value="test_ap">AP ↑</option><option value="test_auroc">AUROC ↑</option><option value="test_rmse">RMSE ↓</option><option value="test_brier">Brier ↓</option></select></label>
        <label class="scenario-control" for="results-scenario">${t('參數情境','Parameter scenario')}<select id="results-scenario"></select></label>
      </div>
      <section id="results-scenario-context" class="scenario-context" aria-labelledby="results-scenario-heading"></section>
      <p id="results-mode-note" class="results-context"></p>
      <figure class="results-figure"><figcaption id="results-chart-title"></figcaption><div class="results-plot" tabindex="0" role="region" aria-label="${t('五模型指標圖，可水平捲動','Five-model metric plot; horizontally scrollable')}"><svg id="results-chart" viewBox="0 0 680 308" role="img" aria-labelledby="results-svg-title results-svg-desc"></svg></div><p id="results-chart-note" class="caption"></p></figure>
      <p id="results-announcement" class="sr-only" role="status" aria-live="polite"></p>
      <div id="results-numbers"></div>
      <p id="results-baselines" class="caption"></p>
      <details class="results-details"><summary>${t('區間、配對差值與時間成本的讀法','Reading intervals, paired differences and runtime')}</summary>
        <p>${t('單一情境的區間從 3 個 seed 平均值重抽樣 1,000 次，取 2.5% 與 97.5% 分位數。只有三個重複，區間是探索性摘要，沒有多重比較校正。跨情境平均不顯示區間，以免把不同生成條件當成同一母體。','Single-scenario intervals resample three seed scores 1,000 times and use the 2.5th and 97.5th percentiles. With only three replicates these are exploratory summaries, without multiple-comparison correction. No interval is shown for the across-scenario mean because generating conditions differ.')}</p>
        <p>${t('ΔAP = 模型 AP − MLP AP。單一情境的 ΔAP 區間以同一 seed 的成對差值計算；跨情境只顯示平均差。區間含 0 時，這三個種子不足以給出清楚的增益方向。表中秒數為訓練、驗證、測試預測與事件評分的平均時間，不含圖建構、bootstrap 與繪圖。','ΔAP = model AP − MLP AP. Single-scenario intervals use differences paired by seed; across scenarios only the mean difference is shown. An interval containing zero does not establish a clear direction of improvement from these three seeds. Seconds cover training, validation, test prediction and event scoring, excluding graph construction, bootstrap and plotting.')}</p>
      </details>
      <details class="results-details"><summary>${t('此情境的可觀測網路指標','Observable network descriptors for this scenario')}</summary><div id="results-descriptors"></div></details>
      <h3 id="sim-selection" tabindex="-1">${t('C. 選模規則能否勝過固定模型？','C. Does selection outperform a fixed model?')}</h3>
      <p>${t('依 3.5 的程序，在留出生成情境評估選模規則。下表與 B 共用圖來源，並同時列出 h=1／3／6；固定模型由開發情境的驗證 AP 選定。','Following 3.5, evaluate selection on held-out generating scenarios. The table shares B’s graph source and lists h=1/3/6 together. The fixed model is chosen using development-scenario validation AP.')}</p>
      <div id="selection-numbers"></div>
      <p class="caption">${t('選模增益 = 規則 AP − 固定模型 AP；regret = 各留出情境的測試最佳 AP − 規則 AP，再取平均。Gray 表示規則預測的前兩名驗證 AP 差距不超過 0.01，不是顯著性檢定。','Selection gain = rule AP − fixed-model AP. Regret averages the gap from each held-out scenario’s test-best AP. Gray means the rule’s two highest predicted validation AP values differ by at most 0.01; it is not a significance test.')}</p>
      <p id="selection-scope"></p>
      <aside class="note"><strong>${t('這輪結果支持到哪裡？','What does this round support?')}</strong><p>${t('目前沒有證據顯示這套指標加淺層決策樹，比固定採用一個在開發情境表現良好的模型更有效。這個結果應保留在報告中；不能把已擬合出規則，等同於已證明規則有效。','This round does not demonstrate that these descriptors plus a shallow tree improve on a fixed model chosen on development scenarios. This result belongs in the report: fitting a rule does not establish that it is useful.')}</p></aside>
      ${heading('3.7','sim-discussion','結果解讀與研究限制','Discussion and limitations',t)}
      <h3>${t('這輪觀察到什麼？','What was observed?')}</h3>
      <p>${t('在 16 個觀測圖設定的等權平均中，Dynamic GCN 的 AP 在三個預測範圍都最高；在 5 個估計圖設定中則是 MLP。兩組設定集合不同，因此應以 3.6 A 的同條件對照，再配合 B 的區間解讀模型適用條件。','Dynamic GCN has the highest mean AP at all three horizons across the 16 observed-graph settings; MLP leads across the five estimated-graph settings. Because these setting sets differ, use the matched comparisons in 3.6 A together with B’s intervals to interpret suitability.')}</p>
      ${scenarios.interpretation(t)}
      <h3>${t('本輪可以支持的結論與後續工作','Supported conclusions and remaining work')}</h3>
      <p>${t('選模規則尚未在留出評估中超過最佳固定模型。這輪也只使用 20 節點、3 seeds、四因素各兩個端點與固定生成器；較密的 100 情境網格尚未執行。後續需要增加 seeds，並檢查圖估計窗、樣本長度、節點數與不同生成機制的敏感度。','The selection rule has not surpassed the best fixed model on held-out evaluation. This round uses 20 nodes, three seeds, two endpoint levels per factor and one generator; the denser 100-setting grid has not run. Further work needs more seeds and sensitivity checks for estimation windows, sequence length, node counts and alternative generators.')}</p>
      <aside class="note"><strong>${t('與真實高濁度預警的界線','Boundary with real turbidity forecasting')}</strong><p>${t('以上全部是合成資料上的 step 預測。現有實測資料在固定 1,000 NTU 門檻與時間切分下，驗證／測試期沒有正例，也缺同期上游水文資料；本頁不把模擬 AP 解讀為石門水庫的預警成功率。','All scores above concern synthetic data and step-based horizons. The current measured dataset has no validation/test positives under the fixed 1,000 NTU threshold and time split, and lacks concurrent upstream covariates. Simulation AP is not a Shihmen early-warning success rate.')}</p></aside>
      <h3 id="sim-downloads" tabindex="-1">${t('結果來源與可重現資料','Provenance and reproducible outputs')}</h3>
      <p>${t('來源為 2026-09-30 完成的 overnight_20260930 批次，945 筆模型結果。圖表直接讀取已核對的結果，參數示意滑桿不會改寫成績。以下檔案可用來重算表格與檢查研究設定。','Source: the overnight_20260930 batch completed on 2026-09-30, with 945 model results. Charts read verified outputs; the illustrative parameter sliders do not change scores. The files below support recomputing tables and auditing the study configuration.')}</p>
      <ul class="results-downloads">${[
        ['data/simulation/metrics.csv','逐 seed 完整指標 · CSV','Per-seed metrics · CSV'],
        ['data/simulation/summary.csv','情境平均與區間 · CSV','Scenario means and intervals · CSV'],
        ['data/simulation/paired_differences.csv','對 MLP 的配對差值 · CSV','Paired differences vs MLP · CSV'],
        ['data/simulation/selection_rule.json','凍結規則與留出評估 · JSON','Frozen rules and holdout evaluation · JSON'],
        ['data/simulation/config.json','本輪完整設定 · JSON','Pilot configuration · JSON'],
        ['data/simulation/provenance.json','版本與檔案校驗碼 · JSON','Version and checksums · JSON']
      ].map(([href,zh,en])=>`<li><a href="${href}" download>${t(zh,en)} ↗</a></li>`).join('')}</ul>
      <p class="caption provenance-hash">${t('演算法版本 SHA-256','Algorithm SHA-256')}: <code>${esc(data.sourceSha256)}</code></p>`;
  }

  function mount(t) {
    const id = key => document.getElementById(key);
    const scenarioOptions = () => {
      const items=scenarios.sorted(state.mode);
      if(state.scenario!=='all'&&!items.some(s=>s.scenario_id===state.scenario))state.scenario='all';
      id('results-scenario').innerHTML=`<option value="all">${t('全部情境等權平均','Equal-weight mean of all scenarios')}</option>`+items.map(s=>`<option value="${s.scenario_id}">${scenarios.code(s)} · ${esc(scenarios.name(s,t))}</option>`).join('');
      id('results-scenario').value=state.scenario;
    };
    id('results-mode').value=state.mode;
    id('results-horizon').value=state.horizon;
    id('results-metric').value=state.metric;
    scenarioOptions();
    id('results-mode').addEventListener('change',e=>{state.mode=e.target.value;state.scenario='all';scenarioOptions();render(t);});
    id('results-horizon').addEventListener('change',e=>{state.horizon=Number(e.target.value);render(t);});
    id('results-scenario').addEventListener('change',e=>{state.scenario=e.target.value;render(t);});
    id('results-metric').addEventListener('change',e=>{state.metric=e.target.value;render(t);});
    document.querySelectorAll('.scenario-guide, #results-matched').forEach(container=>container.addEventListener('click',e=>{
      const button=e.target.closest('[data-scenario-id]');
      if(!button)return;
      const selected=data.scenarios.find(s=>s.scenario_id===button.dataset.scenarioId);
      state.mode=selected.graph_mode;state.scenario=selected.scenario_id;
      id('results-mode').value=state.mode;scenarioOptions();render(t);
      const target=id('results-scenario-heading');
      target.scrollIntoView({behavior:'instant'});
      target.focus({preventScroll:true});
    }));
    document.querySelectorAll('[data-result-jump]').forEach(button=>button.addEventListener('click',()=>{
      const target=id(button.dataset.resultJump);target.scrollIntoView({behavior:'instant'});target.focus({preventScroll:true});
    }));
    render(t);
  }

  function render(t) {
    const id = key => document.getElementById(key);
    const isAll=state.scenario==='all';
    const rows=data.models.map(model=>(isAll?data.aggregate:data.summary).find(r=>r.graph_mode===state.mode&&r.scenario_id===state.scenario&&r.horizon===state.horizon&&r.model===model));
    const scenario=data.scenarios.find(s=>s.scenario_id===state.scenario);
    id('results-scenario-context').innerHTML=scenarios.context(scenario,state.mode,t);
    id('results-matched').innerHTML=scenarios.matched(state.horizon,t);
    const label=`${modeName(state.mode,t)} · h = ${state.horizon} ${t('步','steps')} · ${isAll?t('各情境等權平均','equal-weight scenario mean'):scenarioLabel(scenario)}`;
    id('results-chart-title').textContent=metricName(state.metric)+` · ${['test_ap','test_auroc'].includes(state.metric)?t('越高越好','higher is better'):t('越低越好','lower is better')}`;
    id('results-mode-note').textContent=state.mode==='observed'?t('觀測圖：在潛在圖加入 η 邊翻轉雜訊。固定圖取最初 168 步的多數決；動態圖讀取當下快照。η=0 的設定提供無觀測邊雜訊的參照。','Observed graph: η flips latent edge indicators. Static graphs use the majority graph over the first 168 steps; dynamic graphs read the current snapshot. η=0 supplies a reference without observation-edge noise.'):t('估計圖：只用當時可得序列的滯後偏相關建立 top-k 圖。固定圖只估最初 168 步；動態圖每 6 步以 trailing window 更新。η 不影響輸入，λ=0 時 δ、κ 也無效，因此去重為 5 情境。','Estimated graph: top-k graphs use lagged partial correlations from available series only. Static graphs use the first 168 steps; dynamic graphs update a trailing window every 6 steps. η has no effect, and δ/κ are also ineffective when λ=0, leaving 5 distinct scenarios.');
    id('results-chart-note').textContent=label+t('。 ','. ')+(isAll?t('僅顯示描述性平均；兩種圖建構的情境集合不同，請勿直接當作同一實驗排名。','Descriptive means only. The two arms have different scenario sets and do not form a single controlled ranking.'):t('點為 3 seeds 平均；橫線為跨 seed 的 95% bootstrap 區間。','Points: three-seed means. Lines: 95% bootstrap intervals across seeds.'));
    id('results-announcement').textContent=label+` · ${metricName(state.metric)}`;
    drawPlot(id('results-chart'),rows,state.metric,label,t);
    const mlp=rows[0];
    id('results-numbers').innerHTML=table([t('模型','Model'),'AP','AUROC','RMSE','Brier','ΔAP vs MLP',t('秒','Seconds')],rows.map(r=>{
      const pair=isAll?null:data.paired.find(p=>p.scenario_id===state.scenario&&p.horizon===state.horizon&&p.model===r.model&&p.metric==='test_ap');
      const value=key=>`${fmt(r[key])}${isAll?'':`<small>${ci(r,key)}</small>`}`;
      const gap=r.model==='MLP'?t('基準','Reference'):`${signed(r.test_ap-mlp.test_ap)}${pair?`<small>[${signed(pair.ci_low)}, ${signed(pair.ci_high)}]</small>`:''}`;
      return [esc(r.model),value('test_ap'),value('test_auroc'),value('test_rmse'),value('test_brier'),gap,fmt(r.seconds,1)];
    }),label);
    const base=data.baselines.find(r=>r.graph_mode===state.mode&&r.scenario_id===state.scenario&&r.horizon===state.horizon);
    id('results-baselines').textContent=t(`同一批測試窗：平均正例率 ${fmt(base.prevalence*100,1)}%；訓練盛行率基線 AP ${fmt(base.climatology_ap)}；持續值基線 AP ${fmt(base.persistence_ap)}、RMSE ${fmt(base.persistence_rmse)}。基線亦先逐 seed 計算，再等權平均。`,`Same test windows: mean positive rate ${fmt(base.prevalence*100,1)}%; training-prevalence baseline AP ${fmt(base.climatology_ap)}; persistence AP ${fmt(base.persistence_ap)}, RMSE ${fmt(base.persistence_rmse)}. Baselines are evaluated per seed before equal-weight averaging.`);
    id('results-descriptors').innerHTML=isAll?`<p>${t('請選擇單一參數情境，查看訓練期指標的三種子平均與範圍。這些是觀測量，不是生成參數真值。','Select an individual scenario for training-period descriptor means and ranges across three seeds. These are observable estimates, not latent generating parameters.')}</p>`:table([t('指標','Descriptor'),t('平均','Mean'),t('三種子範圍','Range across seeds')],Object.entries(scenario.descriptors).map(([k,v])=>[esc({graph_utility:'Graph Utility',sdnr:'SDNR',nrc:'NRC',dirichlet_energy:'Dirichlet energy',density:'Density'}[k]),fmt(v.mean),`[${fmt(v.min)}, ${fmt(v.max)}]`]),t('訓練期可觀測指標','Training-period observable descriptors'));
    const rules=data.rules.filter(r=>r.graph_mode===state.mode);
    id('selection-numbers').innerHTML=table(['h',t('規則 AP','Rule AP'),t('最佳固定模型','Best fixed model'),t('固定 AP','Fixed AP'),t('選模增益','Selection gain'),'Regret','Gray'],rules.map(r=>{
      const h=r.heldout;return [String(r.horizon),fmt(h.mean_test_ap),esc(h.best_fixed_model),fmt(h.best_fixed_test_ap),signed(h.mean_test_ap-h.best_fixed_test_ap),fmt(h.mean_selection_regret),fmt(h.gray_fraction*100,1)+'%'];
    }),modeName(state.mode,t)+' · '+t('留出生成情境的測試表現','Test performance on held-out generating scenarios'));
    const held=rules[0].heldout;
    id('selection-scope').textContent=t(`此實驗留出 ${held.heldout_generation_groups.length} 個生成機制群組，涵蓋 ${held.heldout_scenario_ids.length} 個參數情境、每個 horizon 共 ${held.n_heldout_replicates} 筆情境 × seed 評估。相同機制的變體屬於同一群組，不能當成額外的獨立生成機制。`,`This arm holds out ${held.heldout_generation_groups.length} generating-mechanism groups, covering ${held.heldout_scenario_ids.length} parameter scenarios and ${held.n_heldout_replicates} scenario × seed evaluations per horizon. Variants of the same mechanism are grouped together, not counted as additional independent mechanisms.`);
  }

  function drawPlot(svg,rows,key,label,t) {
    const max=['test_ap','test_auroc'].includes(key)?1:Math.ceil(Math.max(...rows.map(r=>r[key+'_ci_high']??r[key]))*1.1*10)/10;
    const left=135,width=450,x=v=>left+width*v/max;
    let content=`<title id="results-svg-title">${esc(metricName(key)+' · '+label)}</title><desc id="results-svg-desc">${esc(t('五模型測試指標；詳細數值與區間見下方表格。','Five-model test metrics; numerical values and intervals are listed below.'))}</desc>`;
    for(let i=0;i<=5;i++){const value=max*i/5,px=x(value);content+=`<line x1="${px}" y1="16" x2="${px}" y2="244" stroke="#d0d5c3"/><text x="${px}" y="273" text-anchor="middle" fill="#596b5d" font-size="13">${fmt(value,2)}</text>`;}
    rows.forEach((r,i)=>{const cy=38+i*45,lo=r[key+'_ci_low'],hi=r[key+'_ci_high'];content+=`<text x="120" y="${cy+5}" text-anchor="end" fill="#243a30" font-size="14">${esc(r.model)}</text>`;
      if(lo!=null&&hi!=null)content+=`<path d="M${x(lo)} ${cy-6}v12 M${x(lo)} ${cy}H${x(hi)} M${x(hi)} ${cy-6}v12" stroke="#708655" stroke-width="2" fill="none"/>`;
      content+=`<circle cx="${x(r[key])}" cy="${cy}" r="5" fill="#315b43"><title>${esc(r.model)}: ${fmt(r[key])}${lo==null?'':` [${fmt(lo)}, ${fmt(hi)}]`}</title></circle><text x="604" y="${cy+5}" fill="#243a30" font-size="14">${fmt(r[key])}</text>`;
    });
    content+=`<text x="360" y="300" text-anchor="middle" fill="#596b5d" font-size="13">${esc(metricName(key))}</text>`;
    svg.innerHTML=content;
  }

  window.SimulationResults={intro,runSummary,methods,selectionMethod,markup,mount};
  window.RESEARCH_DATA.status='pilot_complete';
  window.RESEARCH_DATA.results.simulation=data;
  window.RESEARCH_DATA.results.selectionMap={status:'pilot_evaluated',rules:data.rules};
}());
