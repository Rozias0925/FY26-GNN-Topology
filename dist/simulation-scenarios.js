/* Explain the frozen pilot design; all displayed scores come from its export. */
(function () {
  'use strict';
  const data = window.SIMULATION_RESULTS;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = n => Number(n).toFixed(4);
  const sorted = mode => data.scenarios.filter(s=>s.graph_mode===mode).sort((a,b)=>a.graph_signal-b.graph_signal||a.graph_drift-b.graph_drift||a.edge_noise-b.edge_noise||a.neighbor_heterogeneity-b.neighbor_heterogeneity);
  const code = s => `${s.graph_mode==='observed'?'O':'E'}${String(sorted(s.graph_mode).findIndex(x=>x.scenario_id===s.scenario_id)+1).padStart(2,'0')}`;
  const parameters = s => `λ=${s.graph_signal}, δ=${s.graph_drift}, η=${s.edge_noise}, κ=${s.neighbor_heterogeneity}`;
  const families = [
    {key:'A',signal:0,drift:0,concentration:0,zh:'自身訊號對照',en:'Local-signal control',questionZh:'鄰居完全不參與反應生成時，加入圖聚合是否反而影響表現？',questionEn:'When neighbors do not generate the response, does graph aggregation affect performance?'},
    {key:'B',signal:2,drift:0,concentration:0,zh:'固定關係・均勻影響',en:'Fixed relations, equal influence',questionZh:'有用的鄰居固定、權重相同，GCN 能否從鄰居資訊獲益？',questionEn:'Can GCN benefit when informative neighbors are fixed and equally weighted?'},
    {key:'C',signal:2,drift:0,concentration:3,zh:'固定關係・集中影響',en:'Fixed relations, concentrated influence',questionZh:'少數鄰居比較重要時，GAT 能否學到有用的權重？',questionEn:'Can GAT learn useful weights when a few neighbors matter more?'},
    {key:'D',signal:2,drift:.25,concentration:0,zh:'變動關係・均勻影響',en:'Changing relations, equal influence',questionZh:'有用的鄰居隨時間改變，更新圖是否比固定圖有效？',questionEn:'Does updating the graph help when informative neighbors change over time?'},
    {key:'E',signal:2,drift:.25,concentration:3,zh:'變動關係・集中影響',en:'Changing relations, concentrated influence',questionZh:'關係變動又有權重差異時，動態圖與注意力能否同時帶來增益？',questionEn:'Do dynamic graphs and attention help when both relations and influence differ?'}
  ];
  const family = s => families.find(f=>f.signal===s.graph_signal&&(f.signal===0||f.drift===s.graph_drift&&f.concentration===s.neighbor_heterogeneity));
  const name = (s,t) => t(family(s).zh,family(s).en)+(s.graph_signal===0&&s.graph_mode==='observed'?t(`（圖${s.graph_drift?'變動':'固定'}，κ=${s.neighbor_heterogeneity}）`,` (${s.graph_drift?'changing':'fixed'} graph, κ=${s.neighbor_heterogeneity})`):'')+(s.graph_mode==='observed'?t(s.edge_noise===0?'・無邊雜訊':'・有邊雜訊',s.edge_noise===0?' · no edge noise':' · noisy edges'):'');
  const button = (s,t) => `<button class="scenario-link" data-scenario-id="${esc(s.scenario_id)}" title="${esc(name(s,t)+'; '+parameters(s))}">${code(s)}</button>`;
  const table = (headers,rows,caption,kind='scenario-prose-table') => `<div class="table-wrap ${kind}" tabindex="0" role="region" aria-label="${esc(caption)}"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((c,i)=>`<${i?'td':'th scope="row"'}>${c}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table></div>`;

  function task(t) {
    return `      <p>${t('我們生成一個有 20 個節點的抽象網路。每個節點都有 3 個隨時間變化的外生特徵 x，以及一條待預測的反應序列 r；節點不是已校準的測站，r 也不是實測濁度。資料生成式保留自身歷史的影響，再控制鄰居有沒有用、關係會不會變，以及重要性是否集中。','We generate an abstract network of 20 nodes. Each node has three time-varying exogenous features x and a response series r to predict. These are not calibrated stations, and r is not measured turbidity. The generator retains local-history effects while controlling whether neighbors matter, whether relations change, and whether influence is concentrated.')}</p>
      <div class="comparison-contract"><strong>${t('五模型回答同一個預測問題','One prediction problem for all five models')}</strong><p>${t('給定截至 t 的過去 24 步資料，對每個節點預測：未來 h=1／3／6 步內，r 是否至少一次超過訓練資料第 90 百分位 τ？同時回歸預測 r 在第 t+h 步的值。主比較指標是測試集 AP，並非正確率；合成資料的一步尚未對應到真實的一小時。','Given the previous 24 steps through t, predict for each node whether r exceeds the training-data 90th percentile τ at least once in the next h=1/3/6 steps. Also regress r at t+h. The primary comparison is test AP, not accuracy; a synthetic step has not been calibrated to a real hour.')}</p></div>
`;
  }

  function guide(t) {
    return `<section class="scenario-guide" aria-labelledby="sim-scenarios">
      <h3>${t('由四個參數組成五類研究情境','Five generating families from four factors')}</h3>
      ${table([t('生成情境','Generating scenario'),t('參數與要檢驗的問題','Parameters and research question'),t('對應實驗（點選看結果）','Experiments (select for results)')],families.map(f=>[
        `${f.key}. ${t(f.zh,f.en)}`,
        `<span class="scenario-parameters">λ=${f.signal}${f.signal?`, δ=${f.drift}, κ=${f.concentration}`:''}</span>${t(f.questionZh,f.questionEn)}`,
        ['observed','estimated'].map(mode=>`<div class="scenario-members"><span>${t(mode==='observed'?'觀測圖':'估計圖',mode==='observed'?'Observed':'Estimated')}</span>${sorted(mode).filter(s=>family(s).key===f.key).map(s=>button(s,t)).join(' ')}</div>`).join('')
      ]),t('五類生成情境：情境代表研究問題，不代表預先指定的勝出模型','Five generating families: questions, not preassigned winning models'))}
      <p class="caption">${t('O01–O16 是觀測圖的 2×2×2×2 組合：λ∈{0,2}、δ∈{0,0.25}、η∈{0,0.15}、κ∈{0,3}。λ=0 的 8 組不含鄰居生成項；δ、κ 仍可改變圖或潛在權重，但不改變 r。估計圖不使用 η；λ=0 時 δ、κ 也不改變可見序列，故去重成 E01–E05。21 是實驗設定數，不是 21 種獨立生成機制。','O01–O16 form the observed-graph 2×2×2×2 grid: λ∈{0,2}, δ∈{0,0.25}, η∈{0,0.15}, κ∈{0,3}. The eight λ=0 settings have no neighbor-generating term; δ and κ may change the graph or latent weights, but not r. Estimated graphs do not use η; with λ=0, δ and κ do not change the visible series either, leaving E01–E05 after deduplication. The 21 settings are not 21 independent generating mechanisms.')}</p>
      <details class="results-details"><summary>${t('展開完整的 21 個參數設定','Show all 21 parameter settings')}</summary>${table(['ID',t('情境','Scenario'),'λ','δ','η','κ'],['observed','estimated'].flatMap(mode=>sorted(mode)).map(s=>[button(s,t),esc(name(s,t)),s.graph_signal,s.graph_drift,s.edge_noise,s.neighbor_heterogeneity]),t('ID 僅用於閱讀；原始 scenario_id 保留於下載資料','Display IDs are reading aids; downloads retain original scenario_id values'))}</details>
    </section>`;
  }

  function comparison(t) {
    return `      <h3>${t('兩種圖來源，兩種資訊條件','Two graph sources, two information conditions')}</h3>
      ${table([t('實驗','Arm'),t('固定圖','Static graph'),t('動態圖','Dynamic graph')],[
        [t('觀測圖 O','Observed O'),t('從潛在圖經 η 翻轉邊後的觀測快照，取最初 168 步的多數決圖。','Majority graph over the first 168 observed snapshots, after η flips latent edges.'),t('讀取 t 時刻的觀測快照。η=0 時知道當下連邊，但不知道真實影響權重。','Reads the observed snapshot at t. At η=0 it knows current edges, but not true influence weights.')],
        [t('估計圖 E','Estimated E'),t('只用最初 168 步的 r 序列，依滯後偏相關建立 top-k 圖。','Uses only the first 168 r observations to build a lagged-partial-correlation top-k graph.'),t('以截至 t 的 168 步 r 序列重估，每 6 步更新；max lag=3、top-k=4。','Re-estimates from the trailing 168 r observations every six steps; max lag=3 and top-k=4.')]
      ],t('圖是模型的輸入，不等於模型已知道生成機制','A graph input does not reveal the generating mechanism'))}
      <p class="caption">${t('這裡的 static／dynamic 指上述具體建圖方法。關係快速變動時，最初 168 步的多數決可能留下很少的邊；本輪不能代表所有固定圖方法。估計圖則只從 r 推斷連邊，而生成器的鄰居作用來自 x，因此相關圖未必能恢復有用的鄰居。','Here static/dynamic refer to these particular construction methods. With rapid drift, a majority graph over the first 168 steps may retain few edges; this round does not represent all static-graph methods. Estimated graphs infer edges from r, while the generator’s neighbor effects come from x, so correlation graphs may not recover informative neighbors.')}</p>
      <h3>${t('每個模型究竟看得到什麼？','What information does each model receive?')}</h3>
      ${table([t('模型','Model'),t('可用的預測資訊','Available predictive information'),t('在比較什麼','Comparison being made')],[
        ['MLP',t('只讀取目標節點自己的 24 步 x 與 r 歷史；各節點共用參數，不混合其他節點。','Reads only the target node’s 24-step x and r history. Weights are shared across nodes without mixing them.'),t('自身時間資訊是否已足夠。','Whether local temporal information is sufficient.')],
        ['Static GCN / GAT',t('保留同樣的自身分支，另外透過固定圖聚合鄰居的歷史表示。GCN 使用度數正規化；GAT 學習注意力權重。','Retains the same self branch and aggregates neighbors’ history representations over a fixed graph. GCN uses degree normalization; GAT learns attention weights.'),t('固定的鄰居資訊，以及不同聚合方式的增益。','Benefits of fixed neighbor information and the aggregation method.')],
        ['Dynamic GCN / GAT',t('與對應的 Static 模型架構相同，聚合用的圖換成預測發出時可得的圖。','Uses the corresponding Static architecture, replacing the aggregation graph with the graph available at forecast issue time.'),t('更新圖能否增加預測價值。','Whether updating the graph adds predictive value.')]
      ],t('五模型共用時間編碼器、自身分支與兩個預測頭','All five share a temporal encoder, self branch and two prediction heads'))}
      <p>${t('同一情境、seed 與 h 下，五模型使用相同的生成資料、標籤與時間切分（60%／20%／20%），也有相同的學習率搜尋預算。模型不直接讀取 λ、δ、η、κ 或真實鄰居權重。MLP 與 GCN 各有 7,362 個參數，GAT 為 7,426 個；不能僅以「MLP 參數比較少」解釋排名。','Within a scenario, seed and horizon, all five use the same generated data, labels and chronological split (60%/20%/20%), with equal learning-rate search budgets. They do not read λ, δ, η, κ or true neighbor weights directly. MLP and GCN each have 7,362 parameters; GAT has 7,426. Parameter count alone does not explain the ranking.')}</p>
`;
  }

  function context(s,mode,t) {
    if(!s)return `<h3 id="results-scenario-heading" tabindex="-1">${t('目前是多情境平均，不是單一情境','This is a multi-scenario mean, not one scenario')}</h3><p>${t(mode==='observed'?'正在平均 O01–O16：其中 8 組沒有鄰居生成項、8 組有鄰居項。':'正在平均 E01–E05：1 組沒有鄰居生成項、4 組有鄰居項。',mode==='observed'?'Averaging O01–O16: eight settings without and eight with a neighbor-generating term.':'Averaging E01–E05: one setting without and four with a neighbor-generating term.')} ${t('每個設定先平均 3 個 seeds，再對設定等權平均。要回答哪個模型適合哪種條件，請選上方具名情境，或回到本節開頭的同條件對照表。','Each setting first averages three seeds, then settings receive equal weight. To identify which model suits a condition, select a named scenario above or return to the matched-condition table at the start of this section.')}</p>`;
    return `<h3 id="results-scenario-heading" tabindex="-1">${code(s)} · ${esc(name(s,t))}</h3><p class="scenario-parameters">${parameters(s)}</p><p>${t(family(s).questionZh,family(s).questionEn)}</p><p>${s.graph_signal===0?t('這是 λ=0 對照：反應只由自身上一期反應、自身特徵與隨機創新生成，鄰居項被移除。δ 與 κ 的改變不會改變生成的反應序列。','This λ=0 control generates responses only from the previous local response, local features and random innovations. The neighbor term is removed; δ and κ do not change the generated response series.'):t(`λ=2 保留鄰居作用；δ=${s.graph_drift} 表示${s.graph_drift?'每一步嘗試交換邊，關係隨時間變動':'潛在連邊固定'}；κ=${s.neighbor_heterogeneity} 表示${s.neighbor_heterogeneity?'鄰居影響較集中':'鄰居影響均勻'}。`,`λ=2 retains neighbor effects; δ=${s.graph_drift} ${s.graph_drift?'permits edge swaps at each step':'keeps latent edges fixed'}; κ=${s.neighbor_heterogeneity} gives ${s.neighbor_heterogeneity?'concentrated':'equal'} neighbor influence.`)} ${s.graph_mode==='observed'?t(`η=${s.edge_noise} 是觀測邊翻轉機率，不改變反應序列；${s.edge_noise?'模型可能收到假邊或漏邊':'動態圖可讀到正確連邊'}。`,`η=${s.edge_noise} is the observed-edge flip probability and does not change the response series; ${s.edge_noise?'models may receive false or missing edges':'the dynamic graph reads correct edges'}.`):t('此組僅以反應歷史估計圖，不提供潛在連邊。','This arm estimates graphs from response history without supplying latent edges.')}</p>`;
  }

  function matched(h,t) {
    const rows=[];
    for(const f of families)for(const mode of ['observed','estimated']){
      const s=data.scenarios.find(s=>s.graph_mode===mode&&s.graph_signal===f.signal&&s.graph_drift===f.drift&&s.neighbor_heterogeneity===f.concentration&&s.edge_noise===0);
      const values=data.models.map(model=>data.summary.find(r=>r.scenario_id===s.scenario_id&&r.horizon===h&&r.model===model).test_ap);
      const best=Math.max(...values);
      rows.push([`${f.key}. ${t(f.zh,f.en)}`,`${button(s,t)} ${t(mode==='observed'?'觀測圖':'估計圖',mode==='observed'?'Observed':'Estimated')}`,...values.map(v=>v===best?`<strong>${fmt(v)}</strong>`:fmt(v))]);
    }
    const b=families[1];
    const bscore=(mode,model)=>{const s=data.scenarios.find(s=>s.graph_mode===mode&&s.graph_signal===b.signal&&s.graph_drift===b.drift&&s.neighbor_heterogeneity===b.concentration&&s.edge_noise===0);return data.summary.find(r=>r.scenario_id===s.scenario_id&&r.horizon===h&&r.model===model).test_ap;};
    return `<h3>${t('A. 同一生成條件，不同圖來源','A. Matched conditions, different graph sources')}</h3>
      <p>${t(`下表固定 h=${h}、η=0，列出 A–E 五類情境的觀測圖與估計圖對照。同一類、同一 seed 使用相同的 x、r 與標籤，變更的是提供給 GNN 的圖來源。每個數字為 3 seeds 的測試 AP 平均，粗體僅標示該列最大平均值，不代表差異顯著。`,`With h=${h} and η=0, the table pairs observed and estimated graphs for families A–E. Within a family and seed, x, r and labels are identical; the graph supplied to GNNs changes. Each value is mean test AP over three seeds. Bold marks the largest row mean, not statistical significance.`)}</p>
      <p class="caption">${t('本表固定使用 AP，依本節的預測步數更新。點選 O／E 編號，可在下一部分查看該情境的完整指標與配對差值區間。','This table uses AP and follows this section’s horizon selector. Select an O/E ID to inspect its full metrics and paired-difference intervals in the next part.')}</p>
      ${table([t('同一生成條件','Matched condition'),t('圖來源','Graph source'),...data.models],rows,t(`h=${h} 步 · η=0 · 3 seeds 平均測試 AP`,`h=${h} steps · η=0 · mean test AP over 3 seeds`),'matched-results-table')}
      <p class="matched-reading">${t(`例如 B「固定關係、均勻影響」：MLP 在兩種圖來源皆為 ${fmt(bscore('observed','MLP'))}；Static GCN 從觀測圖的 ${fmt(bscore('observed','Static GCN'))}，變為估計圖的 ${fmt(bscore('estimated','Static GCN'))}。這個比較顯示，模型表現必須連同建圖方法解讀。`,`For B, fixed relations with equal influence, MLP scores ${fmt(bscore('observed','MLP'))} in both arms. Static GCN changes from ${fmt(bscore('observed','Static GCN'))} with observed graphs to ${fmt(bscore('estimated','Static GCN'))} with estimated graphs. Performance must be interpreted together with graph construction.`)}</p>
      `;
  }

  function interpretation(t) {
    return `<h3 id="sim-mlp" tabindex="-1">${t('如何解釋 MLP 在部分條件下領先？','How can MLP’s advantage in some conditions be explained?')}</h3>
      <p>${t('3.6 的對照顯示，排名會隨生成條件與圖來源改變。以下將已知的生成設定，與仍待額外實驗驗證的原因解釋分開討論。','The comparisons in 3.6 show that rankings vary with generating conditions and graph sources. Below, known generator settings are distinguished from explanations that require further experiments.')}</p>
      <ol class="numbered-prose">
        <li><strong>${t('MLP 已經擁有時間資訊。','MLP already has temporal information.')}</strong><p>${t('這裡的 MLP 讀取自身 24 步的 x 與 r。生成式本來就有 0.5rᵢ,ₜ + βᵀxᵢ,ₜ，所以它有可用的預測訊號。λ>0 時，過去的 r 也包含先前鄰居作用留下的資訊；使用 GNN 的關鍵是鄰居能否再增加資訊。','The MLP reads 24 steps of local x and r. The generator includes 0.5rᵢ,ₜ + βᵀxᵢ,ₜ, giving it predictive signal. At λ>0, past r also carries earlier neighbor effects. The question is whether neighbors add information beyond this history.')}</p></li>
        <li><strong>${t('λ=0 本來就是無鄰居作用的對照。','λ=0 deliberately removes neighbor effects.')}</strong><p>${t('此時生成下一步反應不需要其他節點。有限樣本與訓練結果仍可能有波動，但不能先假設圖模型一定要勝出。反過來，λ=2 只是在生成式加入鄰居項，也不保證任何特定 GNN 就能學到它。','The next response is then generated without other nodes. Finite samples and training still produce variation, but there is no reason to require graph models to win. Conversely, λ=2 adds a neighbor term without guaranteeing that a particular GNN can learn it.')}</p></li>
        <li><strong>${t('有用的鄰居，必須能被建圖與聚合方式利用。','Useful neighbors must be accessible to graph construction and aggregation.')}</strong><p>${t('估計圖用 r 的滯後偏相關作為連邊依據，而非直接取得真實連邊；168 步的估計窗也可能平滑掉快速變動的關係。即使連邊正確，模型仍不知道真實的 softmax 鄰居權重。這些是根據程式設定提出的可能解釋；本輪尚未用邊恢復率、不同估計窗或真實權重對照逐項驗證，不能直接斷言「因為圖估錯」或「GNN 過擬合」。','Estimated edges use lagged partial correlations of r instead of true edges; a 168-step window may also smooth rapid changes. Even correct adjacency does not supply the true softmax influence weights. These are plausible explanations based on the implementation, not isolated causal findings. Edge-recovery metrics, window-length comparisons and true-weight controls have not yet tested them; this round cannot establish graph error or GNN overfitting as the cause.')}</p></li>
      </ol>
      <p class="caption">${t('本輪結論的單位是「生成情境 × 建圖方法 × 預測範圍」。目前僅 3 seeds、一種生成器與一組小型架構；下一輪需加入上述對照，才能把觀察到的排名轉為較可靠的原因解釋。','The unit of interpretation is generating condition × construction method × horizon. With only three seeds, one generator and a small architecture family, the controls above are needed before turning rankings into stronger explanations.')}</p>`;
  }

  window.SimulationScenarios={task,guide,comparison,context,matched,interpretation,code,name,sorted};
}());
