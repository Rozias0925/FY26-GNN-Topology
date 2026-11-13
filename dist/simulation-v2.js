/* V2 text follows the frozen mechanism study; results come from a verified export. */
(function () {
  'use strict';
  const models = ['MLP', 'Static-GCN', 'Static-GAT', 'Dynamic-GCN', 'Dynamic-GAT'];
  const inline = tex => `\\(${tex}\\)`;
  // Keep exported field names unchanged; format only their human-readable labels.
  const diagnosticLabels = {S_X:'S_X',S_G:'S_G',H:'H',R_hat:String.raw`\widehat R`,D_struct:String.raw`D_{\mathrm{struct}}`,D_pred:String.raw`D_{\mathrm{pred}}`,A_t:'A_t'};
  const contrastLabels = {
    Architecture_Static:String.raw`\operatorname{GAT}_{S}-\operatorname{GCN}_{S}`,
    Architecture_Dynamic:String.raw`\operatorname{GAT}_{D}-\operatorname{GCN}_{D}`,
    Dynamics_GCN:String.raw`\operatorname{GCN}_{D}-\operatorname{GCN}_{S}`,
    Dynamics_GAT:String.raw`\operatorname{GAT}_{D}-\operatorname{GAT}_{S}`,
    Dynamics_GATv2:String.raw`\operatorname{GATv2}_{D}-\operatorname{GATv2}_{S}`,
    Interaction:'I'
  };
  const modelTex = name => {
    const topology=/^(Static|Dynamic)-(.+)$/.exec(name);
    return topology?String.raw`\mathrm{${topology[2]}}_{${topology[1]==='Static'?'S':'D'}}`:String.raw`\mathrm{${name}}`;
  };
  const displayLabel = value => {
    if (Object.hasOwn(diagnosticLabels,value)) return inline(diagnosticLabels[value]);
    if (Object.hasOwn(contrastLabels,value)) return inline(contrastLabels[value]);
    if (typeof value === 'string' && value.includes('_minus_')) {
      const topology=/^GATv2_minus_GAT_(Static|Dynamic)$/.exec(value);
      if(topology) return inline(String.raw`\mathrm{GATv2}_{${topology[1]==='Static'?'S':'D'}}-\mathrm{GAT}_{${topology[1]==='Static'?'S':'D'}}`);
      return inline(value.split('_minus_').map(modelTex).join('-'));
    }
    return value;
  };
  const parameterSymbols = {graph_signal:String.raw`\lambda`,kappa:String.raw`\kappa`,tau:String.raw`\tau`,delta:String.raw`\delta`,eta:String.raw`\eta`,noise:String.raw`\sigma`,n_nodes:'N',n_steps:'T',degree:'d',phi:String.raw`\phi`};
  const numericToken = value => /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?$/i.test(String(value));
  const texNumber = value => String(value).replace(/e([+-]?\d+)$/i,(_,power)=>String.raw`\times10^{${Number(power)}}`);
  function conditionLabel(row,t) {
    if(row.value==null || row.value==='null' || row.value==='') {
      const regimes={feature_only:t('自身訊號','Feature only'),uniform_neighbors:t('均勻鄰居','Uniform neighbors'),selective_neighbors:t('選擇性鄰居','Selective neighbors'),smooth_signal:t('平滑訊號','Smooth signal'),predictive_dynamics:t('有效動態','Predictive dynamics'),noisy_dynamics:t('雜訊動態','Noisy dynamics')};
      return regimes[row.axis] || '';
    }
    let value=row.value;
    if(typeof value==='string') {try {value=JSON.parse(value);} catch(_) { /* Plain category label. */ }}
    if(value && typeof value==='object' && !Array.isArray(value)) {
      const entries=Object.entries(value);
      if(entries.every(([key,number])=>Object.hasOwn(parameterSymbols,key)&&numericToken(number)))
        return inline(entries.map(([key,number])=>`${parameterSymbols[key]}=${texNumber(number)}`).join(String.raw`,\quad `));
    }
    if(numericToken(value)) {
      const symbol=parameterSymbols[row.axis] || (['classification','architecture_control','query_dependent'].includes(row.axis)?String.raw`\kappa`:'');
      return inline(`${symbol?symbol+'=':''}${texNumber(value)}`);
    }
    if(typeof value==='string') {
      const family=/^(heterogeneous|sbm|knn)_k([\d.]+)$/.exec(value);
      if(family) return ({heterogeneous:t('異質度數圖','Heterogeneous degree'),sbm:'SBM',knn:'kNN'}[family[1]])+' '+inline(String.raw`\kappa=${family[2]}`);
    }
    return String(value);
  }
  const curveOptions = {graph_signal:['圖訊號','Graph signal'],kappa:['鄰居選擇性','Neighbor selectivity'],tau:['平滑訊號','Smooth signal'],delta:['真實拓樸變動','True topology drift'],eta:['觀測圖雜訊','Observed-graph noise']};
  const phaseOptions = {signal_relevance:['圖訊號與鄰居選擇性','Graph signal and neighbor selectivity'],drift_noise:['結構變動與預測增益','Structural change and predictive gain'],smooth_signal:['圖訊號與平滑度','Graph signal and smoothness']};
  const resultSections = [
    ['stage_a','機制檢查','Mechanism checks'],
    ['curves','單軸反應曲線','Response curves'],
    ['interactions','交互作用圖','Interaction maps'],
    ['multi_axis','多軸研究','Multi-axis study'],
    ['sample_size','樣本數比較','Sample size'],
    ['robustness','穩健性檢查','Robustness'],
    ['convergence','訓練預算檢查','Training budget']
  ];
  const resultState={active:'stage_a',curve:'graph_signal',phase:'signal_relevance',robustness:'architecture_control'};
  const robustnessOptions={
    architecture_control:['標準圖基準','Regular-graph controls','robustness_architecture_control_regression.png'],
    graph_family:['圖類型','Graph families','robustness_graph_family_regression.png'],
    noise:['反應雜訊','Response noise','robustness_noise_regression.png'],
    classification:['合成二元分類','Synthetic binary classification','robustness_classification_classification.png'],
    query_dependent:['依查詢而變的鄰居重要性','Query-dependent relevance','robustness_query_dependent_regression.png']
  };
  const curves = {
    graph_signal: ['\\(\\lambda\\) · 圖訊號','\\(\\lambda\\) · Graph signal','S_G','Static / Dynamic：\\(\\operatorname{GCN}/\\operatorname{GAT}-\\operatorname{MLP}\\)','GCN / GAT, static and dynamic, minus MLP','圖的增量資訊增加時，圖模型相對自身特徵模型的收益是否增加？','Does graph-model benefit grow with incremental graph information?'],
    kappa: ['\\(\\kappa\\) · 鄰居選擇性','\\(\\kappa\\) · Neighbor selectivity','\\(R_{\\mathrm{true}}\\) (oracle)','\\(\\operatorname{GAT}-\\operatorname{GCN}\\)（分別比較 Static、Dynamic）','GAT minus GCN, separately for static and dynamic graphs','真正的鄰居重要性越不均勻，注意力相對固定聚合是否更有利？','Does attention gain over fixed aggregation as true relevance becomes unequal?'],
    tau: ['\\(\\tau\\) · 平滑訊號','\\(\\tau\\) · Smooth signal','H','Static / Dynamic：\\(\\operatorname{GCN}/\\operatorname{GAT}-\\operatorname{MLP}\\)','GCN / GAT, static and dynamic, minus MLP','在固定雜訊尺度下，圖平滑是否讓鄰居聚合更有幫助？','Does graph smoothing help aggregation under fixed noise scales?'],
    delta: ['\\(\\delta\\) · 真實拓樸變動','\\(\\delta\\) · True topology drift','D_pred','\\(\\mathrm{Dynamic}-\\mathrm{Static}\\)（分別比較 GCN、GAT）','Dynamic minus static, separately for GCN and GAT','關係的變化帶來預測資訊時，更新圖是否改善表現？','Do updated graphs help when relational changes carry predictive information?'],
    eta: ['\\(\\eta\\) · 觀測圖雜訊','\\(\\eta\\) · Observed-graph noise','\\(D_{\\mathrm{struct}}\\) (\\(D_{\\mathrm{pred}}\\) in tables)','\\(\\mathrm{Dynamic}-\\mathrm{Static}\\)（分別比較 GCN、GAT）','Dynamic minus static, separately for GCN and GAT','只有觀測圖變動、真實關係固定時，動態圖是否追逐了雜訊？','Do dynamic graphs follow noise when only observed edges change?']
  };
  const phaseSpecs = {
    signal_relevance: ['\\(S_G\\times\\widehat R\\)','Architecture_Static','phase_S_G_R_hat_Architecture_Static.png','\\(\\operatorname{GAT}_{S}-\\operatorname{GCN}_{S}\\)；另搭配各圖模型 \\(-\\) MLP 的配對表。','Static-GAT minus Static-GCN; read alongside each graph-model-minus-MLP contrast.'],
    drift_noise: ['\\(D_{\\mathrm{struct}}\\times D_{\\mathrm{pred}}\\)','Dynamics_GCN','phase_D_struct_D_pred_Dynamics_GCN.png','\\(\\operatorname{GCN}_{D}-\\operatorname{GCN}_{S}\\)；另核對 GAT 的同一比較。','Dynamic-GCN minus Static-GCN; check the corresponding GAT contrast too.'],
    smooth_signal: ['\\(S_G\\times H\\)','Static-GCN_minus_MLP','phase_S_G_H_Static-GCN_minus_MLP.png','\\(\\operatorname{GCN}_{S}-\\operatorname{MLP}\\)；區分圖平滑與圖的增量資訊。','Static-GCN minus MLP; distinguish smoothness from incremental graph information.']
  };
  const ready = () => window.MECHANISM_RESULTS?.schema_version === 2 && window.MECHANISM_RESULTS?.status === 'complete';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '—';
  const mean = values => values.length ? values.reduce((a,b)=>a+b,0)/values.length : null;
  function rankCorrelation(pairs) {
    const ranks=values=>values.map(value=>1+values.filter(v=>v<value).length+(values.filter(v=>v===value).length-1)/2);
    const x=ranks(pairs.map(p=>p[0])), y=ranks(pairs.map(p=>p[1]));
    const mx=mean(x), my=mean(y);
    const denominator=Math.sqrt(x.reduce((s,v)=>s+(v-mx)**2,0)*y.reduce((s,v)=>s+(v-my)**2,0));
    return denominator?x.reduce((s,v,i)=>s+(v-mx)*(y[i]-my),0)/denominator:null;
  }
  const group = name => ready() ? window.MECHANISM_RESULTS.groups?.[name] : null;
  const tableHTML = (headers, rows, caption) => `<div class="table-wrap" tabindex="0" role="region" aria-label="${escape(caption)}"><table><caption>${escape(caption)}</caption><thead><tr>${headers.map(h=>`<th scope="col">${escape(displayLabel(h))}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((cell,i)=>`<${i?'td':'th scope="row"'}>${escape(displayLabel(cell))}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  function blank(t, description, footprint='') {
    return `<div class="sim-v2-blank"><strong>${t('等待完整第二版結果接入','Awaiting the complete V2 results')}</strong><p>${description}</p>${footprint?`<small>${escape(footprint)}</small>`:''}</div>`;
  }
  function observation(key,t) {
    const language = t('zh','en');
    const value = window.MECHANISM_DISCUSSION?.[key]?.[language];
    const paragraphs=Array.isArray(value)?value:[value || t('待根據完整的配對差值、區間與診斷指標補入研究觀察；尚未預先指定獲勝模型。','Interpretation will be added after inspecting the complete paired differences, intervals, and diagnostics. No winner is assigned in advance.')];
    const selection=key==='curves'?({graph_signal:0,kappa:0,tau:1,delta:2,eta:3}[resultState.curve]):key==='robustness'?({architecture_control:0,graph_family:0,noise:1,classification:1,query_dependent:2}[resultState.robustness]):null;
    const selected=selection==null?paragraphs:paragraphs.filter((_,i)=>i===selection);
    const remaining=selection==null?[]:paragraphs.filter((_,i)=>i!==selection);
    return `<div class="sim-v2-observation"><strong>${t('研究觀察與解讀','Observations and interpretation')}</strong>${selected.map(text=>`<p>${escape(text)}</p>`).join('')}${remaining.length?`<details class="results-details"><summary>${t('查看其他情境的完整解讀','Read the complete interpretation for other conditions')}</summary>${remaining.map(text=>`<p>${escape(text)}</p>`).join('')}</details>`:''}</div>`;
  }
  function safePath(path) {
    return typeof path === 'string' && /^data\/mechanisms-v2\/[a-z0-9_\-/]+\.(png|csv|json)$/i.test(path) && !path.includes('..');
  }
  function figure(name,filename,caption,t) {
    const path = (group(name)?.figures || []).find(p=>p.split('/').pop()===filename);
    if (!safePath(path)) return blank(t,t('此圖尚未接入；請以同一批次的完整成果更新。','This figure is unavailable. Update from the complete matching batch.'));
    const latexFigures = {
      main: ['response_graph_signal_regression.png','response_kappa_regression.png','response_tau_regression.png','response_delta_regression.png','response_eta_regression.png','phase_S_G_R_hat_Architecture_Static.png','phase_D_struct_D_pred_Dynamics_GCN.png','phase_S_G_H_Static-GCN_minus_MLP.png','sample_size_n_nodes_regression.png'],
      robustness: ['robustness_architecture_control_regression.png','robustness_classification_classification.png','robustness_graph_family_regression.png','robustness_noise_regression.png','robustness_query_dependent_regression.png']
    };
    const displayPath=latexFigures[name]?.includes(filename)?`assets/figures-latex/${name}/${filename}`:path;
    return `<figure><a href="${escape(displayPath)}" data-original-figure="${escape(path)}" target="_blank" rel="noopener"><img src="${escape(displayPath)}" alt="${t('研究結果圖；數學定義與說明見下方圖說。','Research result plot; mathematical definitions and explanations appear in the caption.')}" loading="lazy"></a><figcaption>${caption}</figcaption></figure>`;
  }
  function scoreTable(rows,t,caption) {
    const axes = [...new Set(rows.map(row=>row.axis))];
    const names = {feature_only:t('自身訊號','Feature only'),uniform_neighbors:t('均勻鄰居','Uniform neighbors'),selective_neighbors:t('選擇性鄰居','Selective neighbors'),smooth_signal:t('平滑訊號','Smooth signal'),predictive_dynamics:t('有效動態','Predictive dynamics'),noisy_dynamics:t('雜訊動態','Noisy dynamics')};
    return tableHTML([t('情境','Regime'),...models], axes.map(axis=>[names[axis] || axis,...models.map(model=>number(mean(rows.filter(row=>row.axis===axis&&row.model===model).map(row=>row.test_r2).filter(Number.isFinite))))]),caption);
  }
  function contrastTable(rows,t,caption) {
    const names={Architecture_Static:'Static: \\(\\operatorname{GAT}-\\operatorname{GCN}\\)',Architecture_Dynamic:'Dynamic: \\(\\operatorname{GAT}-\\operatorname{GCN}\\)',Dynamics_GCN:'GCN: \\(\\mathrm{Dynamic}-\\mathrm{Static}\\)',Dynamics_GAT:'GAT: \\(\\mathrm{Dynamic}-\\mathrm{Static}\\)',Interaction:t('\\(I\\)：動態收益的架構交互作用','\\(I\\): architecture interaction in dynamic gain')};
    const label=row=>{
      const contrast=row.contrast==='Interaction'?names.Interaction:displayLabel(row.contrast);
      const condition=conditionLabel(row,t);
      const value=condition?` · ${condition}`:'';
      return contrast+value;
    };
    return tableHTML([t('比較／條件','Contrast / condition'),'\\(\\Delta\\)',t('95% 配對 CI','95% paired CI'),t('正差值種子比例','Positive-seed fraction'),t('種子數','Seeds')],rows.map(row=>[label(row),number(row.mean_gap),`${number(row.ci_low)} – ${number(row.ci_high)}`,number(row.p_delta_positive),row.seeds]),caption);
  }
  function resultsMarkup(t) {
    return `<div class="research-result-tabs" role="tablist" aria-orientation="horizontal" aria-label="${t('模擬結果分頁','Simulation result tabs')}">${resultSections.map(([key,zh,en])=>`<button type="button" role="tab" id="result-tab-${key}" data-result-tab="${key}" aria-controls="sim-result-${key}" aria-selected="${key===resultState.active}" tabindex="${key===resultState.active?'0':'-1'}">${t(zh,en)}</button>`).join('')}</div>
      ${resultSections.map(([key,zh,en],i)=>`<section class="sim-v2-result research-result-panel" role="tabpanel" id="sim-result-${key}" aria-labelledby="result-tab-${key}" tabindex="0"${key===resultState.active?'':' hidden'}><h2 id="sim-result-title-${key}">6.${i+1} ${t(zh,en)}</h2>${panelIntro(key,t)}${key==='curves'?`<label for="v2-curve-axis">${t('選擇單軸實驗','Choose a response curve')}<select id="v2-curve-axis">${Object.keys(curves).map(axis=>`<option value="${axis}"${axis===resultState.curve?' selected':''}>${t(...curveOptions[axis])}</option>`).join('')}</select></label>`:key==='interactions'?`<label for="v2-phase-axis">${t('選擇診斷座標','Choose diagnostic coordinates')}<select id="v2-phase-axis">${Object.keys(phaseSpecs).map(axis=>`<option value="${axis}"${axis===resultState.phase?' selected':''}>${t(...phaseOptions[axis])}</option>`).join('')}</select></label>`:key==='robustness'?`<label for="v2-robustness-axis">${t('選擇穩健性檢查','Choose a robustness check')}<select id="v2-robustness-axis">${Object.entries(robustnessOptions).map(([axis,labels])=>`<option value="${axis}"${axis===resultState.robustness?' selected':''}>${t(labels[0],labels[1])}</option>`).join('')}</select></label>`:''}<div id="v2-content-${key}"></div><div id="v2-observation-${key}"></div></section>`).join('')}
      <details class="results-details research-results-downloads"><summary>${t('資料來源與可重算成果：22 項下載','Sources and reproducible artifacts: 22 downloads')}</summary><div id="v2-downloads"><p>${t('完整匯出提供逐種子分數、配對比較、反應曲線、相圖資料與來源校驗碼。','The complete export provides seed-level scores, paired contrasts, response curves, phase points, and provenance checksums.')}</p></div></details>`;
  }
  function panelIntro(key,t) {
    const copy = {
      stage_a:['先檢查六個極端機制是否產生可解釋的模型行為。表格顯示各模型的平均測試 \\(R^2\\)；單一情境的最高分不能取代反應曲線。','Check whether six extreme mechanisms produce interpretable behavior. The table shows mean test \\(R^2\\); a single winning scenario cannot replace a response curve.'],
      curves:['每條曲線同時對照控制參數與量測指標，顯示平均配對差值及 95% 區間。零線代表平均差異為零；方向不符合假說也必須保留。','Each curve compares the control parameter with a diagnostic and reports mean paired differences with 95% intervals. Zero denotes no average difference. Results that contradict a hypothesis are retained.'],
      interactions:['三組二維實驗把圖訊號、選擇性、平滑度及動態性分開討論。完整相圖合併 C／D 的診斷座標，顏色表示連續分數差，不把邊界硬切成勝者區域。','The interaction grids distinguish signal, selectivity, smoothness, and dynamics. Complete maps combine diagnostic coordinates from C and D and color continuous score gaps rather than imposing winner regions.'],
      multi_axis:['將 200 組多軸條件放進同一診斷空間，檢查單軸觀察在機制同時存在時是否仍一致。此處提供條件涵蓋與診斷範圍；相圖可切換至交互作用分頁查看。','Place 200 multi-axis conditions in the same diagnostic space to check whether one-axis observations persist when mechanisms coexist. This panel summarizes coverage and diagnostic ranges; the interaction tab contains the maps.'],
      sample_size:['固定 \\(\\kappa=0\\)、\\(\\lambda=0.75\\)，改變 \\(N\\in\\{100,300,500,1000\\}\\)，檢查 GCN 與 GAT 的相對表現及種子間變異。節點數增加不是增加獨立資料集種子數。','Fix \\(\\kappa=0\\) and \\(\\lambda=0.75\\), then vary \\(N\\in\\{100,300,500,1000\\}\\) to assess GCN versus GAT and between-seed variation. More nodes do not constitute more independent dataset seeds.'],
      robustness:['分開報告圖類型、反應雜訊、分類任務與 query-dependent relevance。GATv2 是額外對照，主比較仍保留原五模型；分類 AP 與回歸 \\(R^2\\) 不合併平均。','Report graph families, response noise, classification, and query-dependent relevance separately. GATv2 is an additional control; the original five models remain. Classification AP is not averaged with regression \\(R^2\\).'],
      convergence:['以同樣六個 Stage A 情境比較 200／400 epochs 上限；其餘候選、patience 與切分固定。較長預算是敏感度檢查，不能直接宣稱所有模型已收斂。','Compare 200/400-epoch caps on the same six Stage A conditions while retaining candidates, patience, and splits. This is a budget-sensitivity check, not proof that every model converged.']
    };
    return `<p>${t(...copy[key])}</p>`;
  }
  function renderCurve(t) {
    const select=document.getElementById('v2-curve-axis');
    if(!select) return;
    const axis=Object.hasOwn(curves,select.value)?select.value:resultState.curve;
    resultState.curve=axis;
    const s=curves[axis], question=t(s[5],s[6]), gap=t(s[3],s[4]);
    let html=`<h4>${question}</h4><p class="caption">${t('橫軸','Horizontal axes')}: ${t(s[0],s[1])} / ${displayLabel(s[2])} · ${t('縱軸','Vertical axis')}: ${gap}</p>`;
    if (group('main')) {
      html+=figure('main',`response_${axis}_regression.png`,t('配對測試 \\(R^2\\) 差；左圖為控制參數，右圖為診斷指標。\\(\\kappa\\) 的 \\(R_{\\mathrm{true}}\\) 是生成器真值；\\(\\eta\\) 右圖使用 \\(D_{\\mathrm{struct}}\\)，\\(D_{\\mathrm{pred}}\\) 另見表格。\\(\\lambda\\)／\\(\\tau\\) 圖顯示 Static 曲線；固定拓樸下 Dynamic 同架構同分，全部比較另見表格。','Paired test-\\(R^2\\) gaps. Left: control; right: diagnostic. \\(\\kappa\\) uses oracle \\(R_{\\mathrm{true}}\\); \\(\\eta\\) plots \\(D_{\\mathrm{struct}}\\) with \\(D_{\\mathrm{pred}}\\) available in the table. \\(\\lambda\\)/\\(\\tau\\) figures show static curves; identical fixed topology yields matching dynamic scores, with all contrasts in the table.'),t);
      const selected=(group('main').contrasts||[]).filter(row=>row.stage==='B'&&row.axis===axis);
      html+=`<details class="results-details"><summary>${t('查看逐條件配對比較','Inspect condition-level paired contrasts')}</summary>${contrastTable(selected,t,t('單軸實驗配對差值','One-axis paired differences'))}</details>`;
    } else html+=blank(t,t('將呈現各實驗點的平均測試 \\(R^2\\) 差、95% 配對區間，以及控制量與診斷量兩種橫軸。','Each experimental point will show mean test-\\(R^2\\) gaps and 95% paired intervals against both the control and its diagnostic.'),`${t('診斷量','Diagnostic')}: ${displayLabel(s[2])} · 5 paired seeds`);
    const container=document.getElementById('v2-content-curves');
    container.innerHTML=html;
    window.renderReportMath?.(container);
  }
  function renderPhase(t) {
    const select=document.getElementById('v2-phase-axis');
    if(!select) return;
    const axis=Object.hasOwn(phaseSpecs,select.value)?select.value:resultState.phase, s=phaseSpecs[axis];
    resultState.phase=axis;
    let html=`<h4>${s[0]}</h4><p class="caption">${t(s[3],s[4])}</p>`;
    if(group('main')) {
      html+=figure('main',s[2],t('點代表條件，座標是訓練期診斷的跨種子平均，顏色為平均配對差值。這是觀察圖，尚未形成保證性的模型選擇規則。','Points represent conditions; coordinates are mean training-period diagnostics, and colors show mean paired gaps. The map is descriptive, not a guaranteed selection rule.'),t);
      html+=`<details class="results-details"><summary>${t('查看該交互作用網格的配對比較','Inspect paired contrasts for this interaction grid')}</summary>${contrastTable((group('main').contrasts||[]).filter(row=>row.stage==='C'&&row.axis===axis&&row.contrast===s[1]),t,t('Stage C 配對差值','Stage C paired differences'))}</details>`;
    } else html+=blank(t,t('等待 C／D 完整條件後，以可觀測診斷呈現連續表現差。','Waiting for complete C/D conditions to map continuous performance differences against observable diagnostics.'),s[0]);
    const container=document.getElementById('v2-content-interactions');
    container.innerHTML=html;
    window.renderReportMath?.(container);
  }
  function renderStageA(t) {
    const a=group('stage_a');
    document.getElementById('v2-content-stage_a').innerHTML=a?scoreTable(a.model_results||[],t,t('Stage A：跨種子平均測試 \\(R^2\\)','Stage A: mean test \\(R^2\\) across seeds')):blank(t,t('六個極端情境 \\(\\times\\) 五模型。將顯示平均 \\(R^2\\)、各模型相對差異及跨種子不確定性。','Six extreme regimes \\(\\times\\) five models. Mean \\(R^2\\), relative differences, and seed-level uncertainty will be shown.'),'\\(6\\text{ conditions}\\times5\\text{ seeds}\\times5\\text{ models}\\)');
    if(a) document.getElementById('v2-content-stage_a').innerHTML+=`<details class="results-details"><summary>${t('查看 Stage A 配對區間','Inspect Stage A paired intervals')}</summary>${contrastTable(a.contrasts||[],t,t('Stage A 配對差值','Stage A paired gaps'))}</details>`;

  }
  function renderMultiAxis(t) {
    const main=group('main');
    const diagnostics=['S_X','S_G','H','R_hat','D_struct','D_pred'];
    if(main) {
      const rows=(main.model_results||[]).filter(row=>row.stage==='D'&&row.model==='MLP');
      const conditions=new Set(rows.map(row=>row.condition_id));
      document.getElementById('v2-content-multi_axis').innerHTML=`<p>${t('已接入完整條件數','Complete conditions loaded')}: <strong>${conditions.size} / 200</strong></p>`+tableHTML([t('訓練期診斷','Training diagnostic'),t('最小值','Minimum'),t('最大值','Maximum')],diagnostics.map(key=>{const values=rows.map(row=>row[key]).filter(Number.isFinite);return [key,values.length?number(Math.min(...values)):'—',values.length?number(Math.max(...values)):'—'];}),t('Stage D：逐種子診斷範圍','Stage D: diagnostic ranges across seeds'));
      const associations=[['S_G','Dynamic-GCN_minus_MLP'],['S_G','Dynamic-GAT_minus_MLP'],['R_hat','Architecture_Static'],['R_hat','Architecture_Dynamic'],['D_pred','Dynamics_GCN'],['D_pred','Dynamics_GAT']];
      document.getElementById('v2-content-multi_axis').innerHTML+=tableHTML([t('診斷','Diagnostic'),t('配對比較','Paired contrast'),'Spearman \\(\\rho\\)',t('95% CI 全正','95% CI above zero'),t('95% CI 全負','95% CI below zero')],associations.map(([diagnostic,contrast])=>{
        const blocks=(main.contrasts||[]).filter(row=>row.stage==='D'&&row.contrast===contrast);
        return [diagnostic,contrast,number(rankCorrelation(blocks.map(row=>[row[diagnostic],row.mean_gap]))),`${blocks.filter(row=>row.ci_low>0).length} / ${blocks.length}`,`${blocks.filter(row=>row.ci_high<0).length} / ${blocks.length}`];
      }),t('Stage D：條件平均診斷與模型差值的探索性關聯','Stage D: exploratory associations between condition-mean diagnostics and model gaps'))+`<p class="caption">${t('\\(\\rho\\) 使用 200 組條件的跨種子平均；區間計數指各條件的五種子配對 bootstrap。這是多機制共存下的描述性關聯，不等同控制其他因素後的因果效果，也未經多重比較校正。','\\(\\rho\\) uses means across seeds for 200 conditions. Interval counts use five-seed paired bootstraps within each condition. These are descriptive associations under coexisting mechanisms, not effects adjusted for other factors; no multiple-comparison correction is applied.')}</p>`;

      document.getElementById('v2-content-multi_axis').innerHTML+=`<p><button type="button" data-open-result="interactions">${t('查看交互作用相圖','View the interaction maps')}</button></p>`;
    } else {
      document.getElementById('v2-content-multi_axis').innerHTML=blank(t,t('將顯示 200 組條件的診斷涵蓋、表現差與不確定區域。','Will show diagnostic coverage, performance gaps, and uncertain regions for 200 conditions.'),'\\(S_X\\) · \\(S_G\\) · \\(H\\) · \\(\\widehat R\\) · \\(D_{\\mathrm{struct}}\\) · \\(D_{\\mathrm{pred}}\\)');
    }
  }
  function renderSampleSize(t) {
    const main=group('main');
    if(main) {
      document.getElementById('v2-content-sample_size').innerHTML=figure('main','sample_size_n_nodes_regression.png',t('節點數變動時的配對 \\(R^2\\) 差；另分別查看靜態圖與動態圖下的 \\(\\operatorname{GAT}-\\operatorname{GCN}\\) 比較。','Paired \\(R^2\\) gaps across node counts; also inspect \\(\\operatorname{GAT}-\\operatorname{GCN}\\) separately under static and dynamic graphs.'),t)+`<details class="results-details"><summary>${t('查看樣本數的全部配對比較','Inspect all sample-size paired contrasts')}</summary>${contrastTable((main.contrasts||[]).filter(row=>row.stage==='sample_size'),t,t('樣本數配對差值','Sample-size paired gaps'))}</details>`;
    } else {
      document.getElementById('v2-content-sample_size').innerHTML=blank(t,t('將顯示各 \\(N\\) 的 \\(\\operatorname{GAT}-\\operatorname{GCN}\\) 差、區間與五模型原始分數。','Will show GAT-minus-GCN gaps, intervals, and five-model scores at each \\(N\\).'),'\\(N\\in\\{100,300,500,1000\\}\\)');
    }
  }
  function renderRobustness(t) {
 const select=document.getElementById('v2-robustness-axis');
 if(!select) return;
 const axis=Object.hasOwn(robustnessOptions,select.value)?select.value:resultState.robustness;
 resultState.robustness=axis;
 const robust=group('robustness'),labels=robustnessOptions[axis];
 const html=robust?figure('robustness',labels[2],t(String.raw`穩健性情境的配對差；合成 classification 使用 AP，regression 使用 \(R^2\)。`,String.raw`Paired gaps under robustness interventions: AP for synthetic classification, \(R^2\) for regression.`),t)+`<details class="results-details"><summary>${t('查看此類型的配對比較','Inspect paired contrasts for this check')}</summary>${contrastTable((robust.contrasts||[]).filter(row=>row.axis===axis),t,t('穩健性配對差值','Robustness paired gaps'))}</details>`:blank(t,t(String.raw`將分別接入圖類型、不同 \(\sigma\)、分類、GATv2 與 query-dependent relevance 的結果。`,String.raw`Separate results will cover graph families, \(\sigma\), classification, GATv2, and query-dependent relevance.`),'14 conditions · 7 models');
 document.getElementById('v2-content-robustness').innerHTML=html;
}
  function renderConvergence(t) {
    const convergence=group('convergence');
    document.getElementById('v2-content-convergence').innerHTML=convergence?scoreTable(convergence.model_results||[],t,t('400 epochs：跨種子平均測試 \\(R^2\\)','400 epochs: mean test \\(R^2\\) across seeds'))+`<p class="caption">${t('與機制檢查分頁的 200-epoch 表及下載的 budget_sensitivity_paired.csv 一起比較；檢查分數差、達上限比例與假說方向是否改變。','Compare with the 200-epoch table in the Mechanism checks tab and the downloadable budget_sensitivity_paired.csv. Examine score changes, budget-exhaustion flags, and whether contrast directions change.')}</p>`:blank(t,t('將比較同一資料、同一模型的 200／400 epochs 差與訓練上限標記。','Will compare paired 200/400-epoch results and budget-exhaustion flags on identical data.'),'\\(200\\to400\\text{ epochs},\\quad\\mathrm{patience}=25\\)');

    document.getElementById('v2-content-convergence').innerHTML+=`<p><button type="button" data-open-result="stage_a">${t('切換至機制檢查','Open Mechanism checks')}</button></p>`;
  }
  function renderDownloads(t) {
    if(ready()) {
      const downloads=(window.MECHANISM_RESULTS.downloads||[]).filter(d=>safePath(d.path));
      document.getElementById('v2-downloads').innerHTML=`<p>${t('以下檔案來自完整第二版匯出；不同階段與訓練預算分開保存。','These files come from the complete V2 export. Stages and training budgets are kept separate.')}</p><ul class="sim-v2-downloads">${downloads.map(d=>`<li><a href="${escape(d.path)}" download>${escape(d.group)} · ${escape(d.label)}</a></li>`).join('')}</ul>`;
    }

  }
  function renderResult(key,t) {
 const panel=document.getElementById('sim-result-'+key);
 if(!panel) return;
 const renderer={stage_a:renderStageA,curves:renderCurve,interactions:renderPhase,multi_axis:renderMultiAxis,sample_size:renderSampleSize,robustness:renderRobustness,convergence:renderConvergence}[key];
 renderer?.(t);
 const interpretation=document.getElementById('v2-observation-'+key);
 if(interpretation) interpretation.innerHTML=observation(key,t);
 panel.querySelectorAll('[data-open-result]').forEach(button=>button.addEventListener('click',()=>activateResult(button.dataset.openResult,t,true)));
 window.renderReportMath?.(panel);
}
  function activateResult(key,t,focusTab=false) {
 if(!resultSections.some(([name])=>name===key)) return;
 resultState.active=key;
 document.querySelectorAll('[data-result-tab]').forEach(button=>{
  const selected=button.dataset.resultTab===key;
  button.setAttribute('aria-selected',String(selected));
  button.setAttribute('tabindex',selected?'0':'-1');
  if(selected&&focusTab) button.focus();
 });
 document.querySelectorAll('.research-result-panel').forEach(panel=>{panel.hidden=panel.id!=='sim-result-'+key;});
 renderResult(key,t);
}
  function mount(t,chapter='results') {
 document.querySelectorAll('[data-sim-jump]').forEach(button=>button.addEventListener('click',()=>{
  const target=document.getElementById(button.dataset.simJump);
  target?.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
  if(target?.matches('h2')) target.focus({preventScroll:true});
 }));
 if(chapter!=='results'||!document.getElementById('result-tab-stage_a')) return;
 document.querySelectorAll('[data-result-tab]').forEach(button=>{
  button.addEventListener('click',()=>activateResult(button.dataset.resultTab,t));
  button.addEventListener('keydown',event=>{
   const keys=resultSections.map(([key])=>key),current=keys.indexOf(button.dataset.resultTab);
   const next=event.key==='ArrowRight'?(current+1)%keys.length:event.key==='ArrowLeft'?(current-1+keys.length)%keys.length:event.key==='Home'?0:event.key==='End'?keys.length-1:null;
   if(next===null) return;
   event.preventDefault();activateResult(keys[next],t,true);
  });
 });
 for(const [id,property,key]of [['v2-curve-axis','curve','curves'],['v2-phase-axis','phase','interactions'],['v2-robustness-axis','robustness','robustness']]){
  const select=document.getElementById(id);
  if(!select) continue;
  select.value=resultState[property];
  select.addEventListener('change',()=>{resultState[property]=select.value;renderResult(key,t);});
 }
 renderDownloads(t);
 activateResult(resultState.active,t);
}
  window.MechanismResults={ready,mount,getState:()=>({...resultState})};
  function chapterHeader(number,kicker,zh,en,descriptionZh,descriptionEn,t){return `<div class="chapter-heading"><span class="chapter-number">${number} / ${kicker}</span><h1>${t(zh,en)}</h1><p class="subtitle">${t(descriptionZh,descriptionEn)}</p></div>`;}
  function chapterOutline(entries,t){return `<nav class="sim-v2-outline" aria-label="${t('本章目錄','Chapter contents')}"><p>${t('本章閱讀順序','Chapter contents')}</p><ol>${entries.map(([number,id,zh,en])=>`<li><button type="button" data-sim-jump="${id}"><span>${number}</span>${t(zh,en)}</button></li>`).join('')}</ol></nav>`;}
  function chapterView(view,t,p,heading,equation,note,rawTable,pending) {
    const table=(headers,rows,caption)=>rawTable(headers.map(displayLabel),rows.map(row=>row.map(displayLabel)),caption);
    const views={
      generation:()=>`
    ${chapterHeader("04","DATA GENERATION","資料生成","Data Generation","從節點特徵、圖與鄰居權重，逐步建立可控制的合成預測問題。","Build controlled synthetic prediction tasks from local features, graphs, and neighbor weights.",t)}
    ${chapterOutline([["4.1","sim-generation","基本生成器","Base generator"],["4.2","sim-graphs","圖與模型輸入","Graphs and inputs"],["4.3","generation-secondary","其他合成任務","Other synthetic tasks"]],t)}
    ${heading('4.1','合成資料如何生成？','How are the synthetic data generated?','sim-generation')}
    <details class="results-details"><summary>${t('完整固定設定與資料切分','Complete fixed settings and data split')}</summary>
    ${table([t('固定項目','Fixed quantity'),t('正式設定','Formal setting')],[
      [t('節點／時間／特徵','Nodes / times / features'),'\\(N=500,\\quad T=40,\\quad p=8\\)'],
      [t('初始圖','Initial graph'),t('連通、無向、二元 \\(d\\)-regular graph；\\(d=10\\)，共 2,500 條無向邊。','Connected undirected binary \\(d\\)-regular graph; \\(d=10\\), with 2,500 undirected edges.')],
      [t('特徵持續性','Feature persistence'),'\\(\\operatorname{AR}(1),\\quad\\phi=0.6\\)'],
      [t('反應／觀測雜訊','Response / observation noise'),t('\\(\\sigma=0.3\\)；平滑任務的觀測雜訊標準差為 \\(\\operatorname{SD}(\\nu)=0.5\\)。','\\(\\sigma=0.3\\); the smoothness task uses observation-noise standard deviation \\(\\operatorname{SD}(\\nu)=0.5\\).')],
      [t('獨立資料集種子','Independent dataset seeds'),'\\(7,19,31,43,59\\)'],
      [t('任務與切分','Task and split'),t('主實驗單步回歸；按目標時間 60%／20%／20% 切分。','Primary one-step regression; 60%/20%/20% by target time.')]
    ])}
    </details>
    <h3>${t('第一步：分開自身特徵與鄰居內容','Step 1: separate self features and neighbor content')}</h3>
    ${p('每個節點的八維特徵拆成獨立的兩組四維區塊：\\(X^{\\mathrm{self}}\\) 產生自身訊號，\\(X^{\\mathrm{graph}}\\) 產生鄰居內容。兩組以 stationary \\(\\operatorname{AR}(1)\\) 生成，初值與創新項皆為標準高斯；圖、特徵與反應雜訊使用獨立亂數流。五個模型都能讀取自身的全部八維特徵。','Each eight-dimensional feature vector comprises independent four-dimensional self and graph blocks. The self block generates local signal, and the graph block generates neighbor content. Both use stationary \\(\\operatorname{AR}(1)\\) processes with standard Gaussian initial values and innovations. Graph, feature, and response-noise streams are separate. All five models observe all eight local features.')}
    ${equation(String.raw`X_{i,t}=\phi X_{i,t-1}+\sqrt{1-\phi^2}\,\epsilon_{i,t},\qquad \phi=0.6`,'(S1)')}
    ${equation(String.raw`u_{i,t}=f(X^{\mathrm{self}}_{i,t}),\quad f(x)=x_1+0.5x_2^2+\sin(x_3)+0.5x_3x_4`,'(S2)')}
    ${equation(String.raw`r_{j,t}=g(X^{\mathrm{graph}}_{j,t}),\quad g(x)=x_1+0.5x_2^2+0.5\sin(x_3)+0.3x_4`,'(S3)')}
    <h3>${t('第二步：控制鄰居的重要性','Step 2: control neighbor relevance')}</h3>
    ${p('以鄰居自己的特徵產生 relevance score，再在真實鄰居與自身之間做 softmax。\\(\\kappa=0\\) 時，每個來源的重要性相同；\\(\\kappa\\) 增加時，少數來源可能主導。主實驗刻意使用只依賴來源的 ranking，讓 vanilla GAT 有機會表示這個機制。','A relevance score is computed from each source neighbor’s own features, then softmax-normalized across true neighbors and self. At \\(\\kappa=0\\) all sources are equally important; increasing \\(\\kappa\\) can concentrate weight on a few sources. The primary ranking depends only on the source so vanilla GAT can represent it.')}
    ${equation(String.raw`s_{j,t}=b^\top X^{\mathrm{graph}}_{j,t},\quad \alpha^*_{ij,t}=\frac{\exp(\kappa s_{j,t})}{\sum_{k\in\widetilde{\mathcal N}^*_t(i)}\exp(\kappa s_{k,t})},\quad v_{i,t}=\sum_{j\in\widetilde{\mathcal N}^*_t(i)}\alpha^*_{ij,t}r_{j,t}`,'(S4)')}
    ${p('\\(b\\) 是 \\((1,-0.5,0.25,0.5)\\) 的單位化向量；鄰居集合包含 self-loop。乾淨的 \\(d\\)-regular graph 加入 self-loop 後共有 11 個來源，GCN 對稱正規化正好等於均勻聚合。這個對齊不表示所有圖上 GCN 都是算術平均。','\\(b\\) is the unit-normalized vector \\((1,-0.5,0.25,0.5)\\), and neighborhoods include self-loops. The clean \\(d\\)-regular graph has 11 sources after adding self, so symmetric GCN normalization equals uniform aggregation. GCN is not an arithmetic average on arbitrary graphs.')}
    <h3>${t('第三步：混合訊號，產生下一步目標','Step 3: mix signals into the next-step target')}</h3>
    ${p('先只用訓練目標對應的 feature origins，把 \\(u\\)、\\(v\\) 分別標準化為零均值、單位變異，再組合成 \\(Y\\)。\\(\\lambda=0\\) 移除鄰居成分；\\(\\lambda\\) 增加時增加鄰居訊號的設計比重。','Use only feature origins corresponding to training targets to standardize \\(u\\) and \\(v\\) to zero mean and unit variance, then form \\(Y\\). \\(\\lambda=0\\) removes the neighbor component; increasing \\(\\lambda\\) increases its designed weight.')}
    ${equation(String.raw`Y_{i,t+1}=\sqrt{1-\lambda}\,\widetilde u_{i,t}+\sqrt{\lambda}\,\widetilde v_{i,t}+\sigma\epsilon_{i,t},\qquad 0\leq\lambda\leq1`,'(S5)')}
    ${note(t('如何解讀 \\(\\lambda\\)？','How should \\(\\lambda\\) be interpreted?'),'\\(\\lambda\\) 控制兩個標準化訊號成分的設計變異配比，不包含反應雜訊；有限樣本的共變異也可能非零，因此不是總反應的精確解釋變異。鄰居集合包含自身，所以 \\(\\lambda=1\\) 時仍有部分訊號可由自身 \\(X^{\\mathrm{graph}}\\) 取得，不能宣稱 MLP 完全沒有可用訊號。','\\(\\lambda\\) controls the designed variance allocation between standardized signal components, excluding response noise. Finite-sample covariance may be nonzero, so it is not an exact fraction of total explained variance. Self-loops leave some signal accessible from local graph features even at \\(\\lambda=1\\); MLP is not necessarily information-free.')}


    ${heading('4.2','真實圖、觀測圖與五模型輸入','True graphs, observed graphs, and model inputs','sim-graphs')}
    ${table([t('符號／名稱','Symbol / object'),t('來源與用途','Source and role')],[
      ['\\(A_t^*\\)',t('生成器的真實二元拓樸，用於產生 \\(Y\\)；作為已知生成機制，與提供模型的觀測拓樸分開記錄。','True binary generating topology used to form \\(Y\\); recorded as a known generating mechanism separately from the observed topology supplied to models.')],
      ['A_t',t('提供模型的觀測二元拓樸；無雜訊時等於 \\(A_t^*\\)，否則由保度數邊交換得到。','Observed binary topology supplied to models; equals \\(A_t^*\\) without corruption, otherwise uses degree-preserving swaps.')],
      ['\\(\\alpha_{ij,t}^*\\)',t('生成 \\(Y\\) 的實數鄰居權重；不提供給 GCN／GAT。GAT 必須從訓練資料學習權重。','Real-valued generating relevance weights; hidden from GCN/GAT. GAT learns its weights from training data.')],
      ['\\(A^{\\mathrm{Static}}\\)',t('只用 \\(t\\le24\\) 的觀測圖，按邊出現頻率選最高的 2,500 條邊。','Select the 2,500 most frequent observed edges using only snapshots at \\(t\\le24\\).')]
    ])}
    ${p('第二版主機制研究直接觀察生成拓樸，再分離真實變動與觀測誤差；本章不從反應序列估計相似度矩陣，因此結論不涵蓋這類統計建圖方法的估計誤差。Static consensus 與動態圖保持相同邊數，但 consensus 不保證仍是 \\(d\\)-regular。','The primary V2 study observes generated topology and separates true change from observation error. It does not estimate a similarity matrix from response histories, so its conclusions do not cover estimation error from such statistical graph-construction methods. Static consensus has the same edge count as dynamic snapshots but need not remain \\(d\\)-regular.')}
    ${table([t('模型','Model'),t('時間 \\(t\\) 可讀取的資訊','Inputs available at time \\(t\\)'),t('聚合方式','Aggregation')],[
      ['MLP',t('自身八維 \\(X_{i,t}\\)。','All eight local features \\(X_{i,t}\\).'),t('自身 nonlinear encoder 與 readout。','Local nonlinear encoder and readout.')],
      ['Static-GCN',t('自身特徵、\\(A^{\\mathrm{Static}}\\) 允許的來源。','Local features and sources allowed by \\(A^{\\mathrm{Static}}\\).'),t('固定拓樸；對稱正規化聚合。','Fixed topology; symmetric normalized aggregation.')],
      ['Static-GAT',t('自身特徵、\\(A^{\\mathrm{Static}}\\) 允許的來源。','Local features and sources allowed by \\(A^{\\mathrm{Static}}\\).'),t('固定拓樸；attention 隨特徵更新。','Fixed topology; attention changes with features.')],
      ['Dynamic-GCN',t('自身特徵、當下 \\(A_t\\) 允許的來源。','Local features and sources allowed by current \\(A_t\\).'),t('每一期使用當下拓樸。','Use the current snapshot topology.')],
      ['Dynamic-GAT',t('自身特徵、當下 \\(A_t\\) 允許的來源。','Local features and sources allowed by current \\(A_t\\).'),t('拓樸與特徵相關的 attention 均可變。','Topology and feature-dependent attention may both change.')]
    ])}
    ${p('這裡的 Dynamic 是 snapshot-dynamic：模型使用當下 \\(X_t\\) 與 \\(A_t\\)，沒有 RNN、記憶狀態、節點 ID 或反應歷史。Static-GAT 的固定 topology，與文獻中 vanilla GAT 的 query-independent「static attention」是不同概念。','Dynamic here means snapshot-dynamic: the model uses current \\(X_t\\) and \\(A_t\\) without an RNN, memory state, node IDs, or response history. Static-GAT’s fixed topology is distinct from vanilla GAT’s query-independent “static attention” limitation.')}


    ${heading("4.3","平滑、拓樸變動與合成分類","Smoothing, topology changes, and synthetic classification","generation-secondary")}
    ${p("主目標是單步連續回歸。獨立平滑實驗改為同時點去雜訊；合成二元分類用於檢查任務穩健性。以下定義將三種任務的標籤與時間對齊分開。","The primary target is one-step continuous regression. Isolated smoothing uses same-time denoising, while synthetic binary classification tests task robustness. The definitions below distinguish labels and temporal alignment across these tasks.")}
    <details class="results-details"><summary>${t('平滑任務與拓樸變動的數學定義','Mathematical definitions of smoothing and topology change')}</summary>
      <h3>${t('平滑度：保持雜訊尺度可比較','Smoothness: keep noise scales comparable')}</h3>
      ${equation(String.raw`z_\tau=(I+\tau L)^{-1}z_0,\qquad z_0\sim\mathcal N(0,I)`,'(S6)')}
      ${p('Stage A 的平滑訊號情境與 Stage B 的 \\(\\tau\\) 實驗把 \\(z_\\tau\\) 用訓練統計標準化，再提供 \\(X_{i,t,1}=\\widetilde z_{\\tau,i,t}+\\nu_{i,t}\\)，其餘七維為干擾特徵，目標 \\(Y_{i,t}=\\widetilde z_{\\tau,i,t}+\\sigma\\epsilon_{i,t}\\)。這是同時點 denoising，使用 1–24／25–32／33–40 切分；不是一步 forecasting。固定觀測雜訊標準差 \\(\\operatorname{SD}(\\nu)=0.5\\) 與反應雜訊尺度 \\(\\sigma=0.3\\)，避免 \\(\\tau\\) 只因改變尺度而混入 SNR 效果。','For the Stage A smooth-signal regime and the Stage B \\(\\tau\\) axis, \\(z_\\tau\\) is standardized using training moments. \\(X_{i,t,1}=\\widetilde z_{\\tau,i,t}+\\nu_{i,t}\\), with seven distractor features, predicts \\(Y_{i,t}=\\widetilde z_{\\tau,i,t}+\\sigma\\epsilon_{i,t}\\). This is same-time denoising with splits 1–24/25–32/33–40, not forecasting. Observation-noise standard deviation \\(\\operatorname{SD}(\\nu)=0.5\\) and response-noise scale \\(\\sigma=0.3\\) remain fixed to reduce SNR confounding.')}
      <h3>${t('真實漂移與觀測雜訊分開','Separate true drift from observation noise')}</h3>
      ${p('\\(\\delta\\) 控制前一期真實圖中被替換的原邊比例；\\(\\eta\\) 控制當下真實圖的觀測腐化比例。兩者都使用 double-edge swaps，維持度數、邊數與連通性，並記錄實際替換率。\\(\\eta\\) 的誤差每期獨立生成，不回寫真實圖與目標；觀測圖的變動因此可能沒有正的預測增益。','\\(\\delta\\) controls the proportion of original edges replaced in the preceding true graph. \\(\\eta\\) controls corruption of the current true graph. Both use double-edge swaps, retaining degrees, edge count, and connectivity while recording realized turnover. Independent observation errors do not alter the true graph or targets, so observed changes may bring no positive predictive gain.')}
    </details>
  
    <h3>${t('合成二元分類標籤','Synthetic binary classification label')}</h3>
    ${p("以不含反應雜訊的連續目標 \\(Y^*\\) 定義分類標籤；門檻固定為其訓練期第 80 百分位。驗證與測試沿用同一門檻，分類分數不與回歸分數合併。","Classification thresholds the noiseless continuous target \\(Y^*\\) at its training 80th percentile. Validation and test use that same threshold, and classification scores are not pooled with regression scores.")}
    ${equation("Y^{\\mathrm{binary}}=\\mathbf1\\{Y^*>c_{\\mathrm{train},0.8}\\}","(S11)")}
    <p><a href="#design">${t('接著閱讀：實驗設計、診斷與評估','Continue: experimental design, diagnostics, and evaluation')} →</a></p>
      `,
      design:()=>`
    ${chapterHeader("05","EXPERIMENTS & EVALUATION","實驗與評估","Experiments & Evaluation","先隔離資料機制，再以相同資料、診斷與配對推論比較五模型。","Isolate data mechanisms, then compare five models using shared data, diagnostics, and paired inference.",t)}
    ${chapterOutline([["5.1","sim-question","研究問題","Research questions"],["5.2","sim-design","實驗情境","Experimental scenarios"],["5.3","sim-diagnostics","可量測診斷","Measurable diagnostics"],["5.4","sim-evaluation","公平比較","Fair comparison"]],t)}
    ${heading('5.1','研究問題與方向性假說','Research questions and directional hypotheses','sim-question')}
    ${table([t('問題','Question'),t('比較','Comparison'),t('預期的機制反應','Directional hypothesis')],[
      [t('圖是否提供額外資訊？','Does a graph add information?'),'\\(\\operatorname{GCN}/\\operatorname{GAT}-\\operatorname{MLP}\\)',t('\\(S_G\\) 增加時，圖模型相對 MLP 的收益可能增加。','Graph-model gains over MLP may grow with \\(S_G\\).')],
      [t('鄰居是否同樣重要？','Are neighbors equally relevant?'),'\\(\\operatorname{GAT}-\\operatorname{GCN}\\)',t('鄰居重要性更集中時，選擇性聚合的相對收益可能增加。','Selective aggregation may help as relevance becomes concentrated.')],
      [t('拓樸變化是否有預測價值？','Is topology change predictive?'),'\\(\\mathrm{Dynamic}-\\mathrm{Static}\\)',t('\\(D_{\\mathrm{pred}}\\) 增加時動態圖可能有利；只有結構變動並不足夠。','Dynamic graphs may help as \\(D_{\\mathrm{pred}}\\) grows; structural change alone is insufficient.')]
    ])}
    ${p('這些是方向性假說，不是預設答案。\\(\\lambda=0\\) 時 MLP 應具有競爭力；均勻鄰居機制允許 GAT 學到接近 GCN 的聚合，因此也不要求 GCN 大勝。如果曲線沒有預期方向，我們檢查資料診斷、模型能力與訓練限制，保留結果，不把生成器改到某個模型勝出。','These are hypotheses, not prescribed outcomes. MLP should be competitive at \\(\\lambda=0\\). GAT can learn nearly uniform aggregation, so a uniform-neighbor mechanism does not require a large GCN victory. Unexpected curves prompt analysis of diagnostics, capacity, and training limits; the generator is not changed to manufacture a winner.')}


    ${heading('5.2','實驗情境：一次隔離一個機制，再研究交互作用','Experimental design: isolate mechanisms, then combine them','sim-design')}
    <ol class="sim-v2-stages"><li><div><h3>Stage A · ${t('六個極端機制','Six mechanism checks')}</h3>${p('依序比較自身訊號（\\(\\lambda=0\\)）、均勻鄰居（\\(\\lambda=0.9\\)、\\(\\kappa=0\\)）、選擇性鄰居（\\(\\lambda=0.9\\)、\\(\\kappa=4\\)）、平滑訊號（\\(\\tau=2\\)）、有效動態（\\(\\lambda=0.9\\)、\\(\\delta=0.4\\)）與雜訊動態（\\(\\lambda=0.9\\)、\\(\\eta=0.4\\)）。檢查生成器是否真的產生要研究的機制，不用模型勝負作為通關門檻。','Compare feature-only (\\(\\lambda=0\\)), uniform neighbors (\\(\\lambda=0.9\\), \\(\\kappa=0\\)), selective neighbors (\\(\\lambda=0.9\\), \\(\\kappa=4\\)), smooth signal (\\(\\tau=2\\)), predictive dynamics (\\(\\lambda=0.9\\), \\(\\delta=0.4\\)), and noisy dynamics (\\(\\lambda=0.9\\), \\(\\eta=0.4\\)). These check generating mechanisms, with no required winning model.')}</div></li><li><div><h3>Stage B · ${t('五條單軸反應曲線','Five one-axis response curves')}</h3>${p('共 28 個實驗點。除平滑任務外，基準 \\(\\lambda=0.75\\)、\\(\\kappa=\\delta=\\eta=\\tau=0\\)；每條軸只改其指定參數，其餘保持基準。用表現差的變化，而非某一個最高分，檢驗假說。','There are 28 experimental references. Outside the smoothing task, the baseline is \\(\\lambda=0.75\\) and \\(\\kappa=\\delta=\\eta=\\tau=0\\). Each axis changes only its designated parameter. Hypotheses are assessed through changing performance gaps rather than individual highest scores.')}</div></li></ol>
    <details class="results-details"><summary>${t('展開 Stage B 五軸取值與完整比較','Expand Stage B axis levels and full comparisons')}</summary>
    ${table([t('軸／取值','Axis / values'),t('產生的機制','Mechanism'),t('主要比較與診斷','Primary contrast / diagnostic')],[
      ['\\(\\lambda\\in\\{0,0.1,0.25,0.5,0.75,0.9,1\\}\\)',t('逐漸增加鄰居成分。','Increase the neighbor component.'),'Each \\(\\operatorname{GNN}-\\operatorname{MLP}\\) · \\(S_G\\)'],
      ['\\(\\kappa\\in\\{0,0.5,1,2,4\\}\\)',t('真實圖固定；來源重要性由均勻變集中。','Keep true topology fixed; concentrate relevance.'),'\\(\\operatorname{GAT}-\\operatorname{GCN}\\) · \\(R_{\\mathrm{true}}\\) / \\(\\widehat R\\)'],
      ['\\(\\tau\\in\\{0,0.1,0.5,1,2,5\\}\\)',t('獨立的平滑訊號去雜訊任務。','Separate smooth-signal denoising task.'),'Each \\(\\operatorname{GNN}-\\operatorname{MLP}\\) · \\(H\\)'],
      ['\\(\\delta\\in\\{0,0.05,0.1,0.2,0.4\\}\\)',t('真實圖逐期保度數重接；\\(\\eta=0\\)。目標使用當下真實圖。','Rewire true topology over time, with \\(\\eta=0\\). Targets use the current true graph.'),'\\(\\mathrm{Dynamic}-\\mathrm{Static}\\) · \\(D_{\\mathrm{pred}}\\)'],
      ['\\(\\eta\\in\\{0,0.05,0.1,0.2,0.4\\}\\)',t('真實圖固定（\\(\\delta=0\\)）；每期觀測圖獨立加入重接誤差。','Fix true topology (\\(\\delta=0\\)); independently corrupt each observed snapshot.'),'\\(\\mathrm{Dynamic}-\\mathrm{Static}\\) · \\(D_{\\mathrm{struct}}\\) / \\(D_{\\mathrm{pred}}\\)']
    ])}
    </details>
    <p>${t('平滑、真實拓樸變動與觀測雜訊的生成定義見','Generating definitions for smoothing, true drift, and observation noise are in')} <a href="#generation">${t('資料生成章','Data Generation')}</a>。</p>
  <ol class="sim-v2-stages" style="counter-reset:stages 2"><li><div><h3>Stage C · ${t('三組二維交互作用','Three interaction grids')}</h3>${p('在單軸機制已交代清楚後，再組合兩個參數，共 56 個實驗引用點。圖有資訊、圖平滑、鄰居選擇性和有用的動態性是不同概念，不能用一個參數代替全部。','Combine two controls only after explaining the one-axis mechanisms: 56 experimental references. Incremental graph information, smoothness, selectivity, and useful dynamics are distinct concepts.')}</div></li></ol>
    <details class="results-details"><summary>${t('展開 Stage C 完整參數網格','Expand the complete Stage C parameter grids')}</summary>
    ${table([t('參數網格','Parameter grid'),t('取值','Levels'),t('診斷座標與問題','Diagnostic space and question')],[
      ['\\(\\lambda\\times\\kappa\\)','\\(\\{0,0.25,0.5,0.75,1\\}\\times\\{0,1,2,4\\}\\quad(20)\\)',t('\\(S_G\\times\\widehat R\\)：圖的增量資訊與鄰居選擇性如何共同影響模型？','\\(S_G\\times\\widehat R\\): how do incremental information and selectivity interact?')],
      ['\\(\\delta\\times\\eta\\)','\\(\\{0,0.1,0.2,0.4\\}\\times\\{0,0.1,0.2,0.4\\}\\quad(16)\\)',t('\\(D_{\\mathrm{struct}}\\times D_{\\mathrm{pred}}\\)：圖改變多少，與改變是否有用。','\\(D_{\\mathrm{struct}}\\times D_{\\mathrm{pred}}\\): how much the graph changes versus whether change is useful.')],
      ['\\(\\tau\\times\\lambda\\)','\\(\\{0,0.5,2,5\\}\\times\\{0,0.25,0.5,0.75,1\\}\\quad(20)\\)',t('\\(S_G\\times H\\)：圖平滑不等於圖具有增量資訊。','\\(S_G\\times H\\): smoothness is not the same as incremental information.')]
    ])}
    </details>
    ${p('Stage C／D 的 \\(\\tau\\) 使用另一個明確延伸：平滑 \\(X^{\\mathrm{graph}}\\) 的四維區塊，再由原本混合機制產生下一步目標；\\(X^{\\mathrm{self}}\\) 保持獨立。它不等同 Stage B 的獨立 denoising 任務，結果必須分開解讀。','\\(\\tau\\) in Stages C/D uses a distinct extension: smooth the four graph-feature dimensions, then generate next-step targets through the mixture mechanism, while keeping the self block independent. This is not the isolated Stage B denoising task and must be interpreted separately.')}
    <ol class="sim-v2-stages" style="counter-reset:stages 3"><li><div><h3>Stage D · ${t('200 組多軸條件','200 multi-axis conditions')}</h3>${p('使用 Latin hypercube sampling，從 \\(\\lambda\\in[0,1]\\)、\\(\\kappa\\in[0,4]\\)、\\(\\delta,\\eta\\in[0,0.4]\\)、\\(\\tau\\in[0,2]\\) 取得 200 組條件，設計種子 \\(\\mathrm{seed}_{\\mathrm{design}}=20261002\\)；其餘正式設定固定。每組用五個獨立資料種子，避免完整 factorial grid 爆炸，同時檢查多機制共同存在時的模型行為。','Latin hypercube sampling draws 200 conditions from \\(\\lambda\\in[0,1]\\), \\(\\kappa\\in[0,4]\\), \\(\\delta,\\eta\\in[0,0.4]\\), and \\(\\tau\\in[0,2]\\), with \\(\\mathrm{seed}_{\\mathrm{design}}=20261002\\) and other formal settings fixed. Each condition uses five dataset seeds, testing coexistence without a full factorial grid.')}</div></li><li><div><h3>${t('補充：樣本量、穩健性與訓練預算','Additional checks: sample size, robustness, and budget')}</h3>${p('樣本數使用 \\(N\\in\\{100,300,500,1000\\}\\)。14 組穩健性情境包括 regular controls、異質度數圖、SBM、kNN、不同 \\(\\sigma\\)、二元分類與 query-dependent relevance，額外加入 Static／Dynamic-GATv2。另將六組 Stage A 的 max epochs 從 200 提高到 400，分開保存預算敏感度結果。','Sample-size checks use \\(N\\in\\{100,300,500,1000\\}\\). Fourteen robustness conditions include regular controls, heterogeneous-degree graphs, SBM, kNN, different \\(\\sigma\\), binary classification, and query-dependent relevance, with Static/Dynamic-GATv2 added. The six Stage A regimes are also rerun with the epoch cap raised from 200 to 400, stored as separate sensitivity results.')}</div></li></ol>
    ${p('B／C／sample-size 中相同條件只計算一次，仍保留各實驗引用。主佇列為 264 個唯一條件、6,600 次評估；加上 150 次預算檢查與 490 次穩健性評估，共 7,240 次。原 Stage A 的 150 次 200-epoch 評估另行保存。完全相同拓樸下，同架構的 Static／Dynamic 可引用相同訓練成果，來源會明確記錄；評估數不等於實際 optimizer 訓練次數。','Identical B/C/sample-size conditions are computed once while retaining experiment references. The primary queue contains 264 unique conditions and 6,600 evaluations; 150 budget-check and 490 robustness evaluations bring the total to 7,240. The original 150 Stage A evaluations at 200 epochs are separate. Identical topology permits explicit same-architecture static/dynamic reuse, so evaluation counts are not actual optimizer-fit counts.')}


    ${heading('5.3','從模擬參數轉為可量測資料特徵','From controls to measurable data diagnostics','sim-diagnostics')}
    ${p('\\(\\lambda\\)、\\(\\kappa\\)、\\(\\tau\\)、\\(\\delta\\)、\\(\\eta\\) 是生成器控制量；診斷量則只由訓練資料估計，用來檢查可觀測特徵能否描述模型表現，不能直接把設定值當成量測結果。\\(S_X\\)、\\(S_G\\)、\\(D_{\\mathrm{pred}}\\) 以固定 probe 在訓練期內做三折 expanding chronological validation；不讀主驗證期或測試期。\\(R_{\\mathrm{true}}\\) 用於核對生成機制，診斷相圖使用估計的 \\(\\widehat R\\)，並檢查兩者的誤差。','\\(\\lambda\\), \\(\\kappa\\), \\(\\tau\\), \\(\\delta\\), and \\(\\eta\\) are generating controls. Diagnostics are estimated from training data to assess whether observable characteristics describe model behavior; configured values are not treated as measurements. \\(S_X\\), \\(S_G\\), and \\(D_{\\mathrm{pred}}\\) use fixed probes with three expanding chronological folds within training, without main validation/test data. \\(R_{\\mathrm{true}}\\) checks the generating mechanism, while diagnostic maps use estimated \\(\\widehat R\\) and examine its estimation error.')}
    ${table([t('指標','Diagnostic'),t('要量測什麼','What does it measure?')],[
      ['S_X',t('自身特徵的預測訊號。','Predictive signal in local features.')],
      ['S_G',t('鄰居摘要提供的增量資訊。','Incremental information from neighbor summaries.')],
      ['H',t('目標相對隨機指派的圖平滑度。','Target smoothness relative to random assignment.')],
      ['R_hat',t('鄰居重要性集中程度，並與生成真值對照。','Concentration of neighbor relevance, checked against generating truth.')],
      ['D_struct',t('相鄰觀測拓樸改變多少。','How much adjacent observed topologies change.')],
      ['D_pred',t('當下圖比共識圖多出多少預測資訊。','Predictive information from current versus consensus graphs.')]
    ])}
    <details class="results-details"><summary>${t('診斷公式、估計方法與負對照','Diagnostic formulas, estimation, and negative controls')}</summary>
    ${table([t('診斷','Diagnostic'),t('操作定義','Operational definition'),t('解釋與限制','Meaning and limitation')],[
      ['S_X','\\(R^2(\\text{feature-only probe})\\)',t('自身八維特徵可被固定 probe 擷取的預測訊號。可為負。','Predictive signal captured by a fixed local-feature probe. Can be negative.')],
      ['S_G','\\(R^2(\\text{features + dynamic neighbor summary})-R^2(\\text{features})\\)',t('相同 probe 加入鄰居平均後的增量收益；不是所有圖預測器的資訊總量。','Incremental gain from a uniform neighbor summary; not total information available to every graph predictor.')],
      ['H','\\(1-\\frac{E(Y)}{\\mathbb E_\\pi[E(\\pi(Y))]}\\)',t('相較隨機指派是否更平滑；\\(H>0\\) 表示較平滑，\\(H\\) 也可為負。','Smoothness relative to random assignment. \\(H>0\\) means smoother; negative values are possible.')],
      ['\\(R_{\\mathrm{true}}\\) / \\(\\widehat R\\)','\\(1-\\text{normalized attention entropy}\\)',t('重要性集中程度；前者是生成真值，後者由訓練期 edge-gating probe 估計。','Concentration of relevance: generating truth versus a training-period edge-gating estimate.')],
      ['D_struct','mean adjacent-snapshot Jaccard distance',t('觀測圖改變多少；不代表變動有用。','Amount of observed topology change; not its usefulness.')],
      ['D_pred','\\(R^2(\\text{dynamic-summary probe})-R^2(\\text{static-summary probe})\\)',t('當下圖相對訓練共識圖的預測增益；純雜訊時可為負。','Predictive gain from current versus training-consensus graphs. Can be negative under noise.')]
    ])}
      ${p('Probe 使用固定的 quadratic 與 \\(\\sin\\) 特徵基底、Ridge \\(\\alpha=10\\)；鄰居摘要是包含 self-loop 的同一基底均勻平均。每折只在較早的訓練時間 fit preprocessing 與 probe，合併後續的 out-of-fold predictions 計算 \\(R^2\\)；每折 static consensus 也只用該折的 fit snapshots。','The probe uses a fixed quadratic-and-sine basis and Ridge \\(\\alpha=10\\). Neighbor summaries uniformly average the same basis including self. Each fold fits preprocessing and the probe on earlier training times, pooling subsequent out-of-fold predictions for \\(R^2\\). Each fold’s static consensus uses only its fit snapshots.')}
      ${p('負對照以固定節點 permutation 重標觀測圖，保留 density、degree 與時間結構，再計算 \\(S_{G,\\mathrm{shuffled}}\\)。真正圖與打亂圖的增量收益若相近，就缺乏圖具有特定任務資訊的證據。高 \\(\\kappa\\) 下，uniform-summary probe 也可能低估選擇性訊號。','The shuffled-graph control relabels observed graphs with a fixed node permutation, preserving density, degree, and temporal structure, then computes \\(S_{G,\\mathrm{shuffled}}\\). Similar gains from true and shuffled graphs weaken evidence of task-specific information. Uniform-summary probes may also miss selective signal at high \\(\\kappa\\).')}
      ${equation(String.raw`E(Y)=Y^\top LY=\sum_{\{i,j\}\in E}(Y_i-Y_j)^2,\qquad H=1-\frac{\sum_t E_t(Y_t)}{\sum_t\mathbb E_\pi[E_t(\pi(Y_t))]}`,'(S7)')}
      ${p('\\(H\\) 使用觀測圖與對齊的目標；每張快照的 permutation expectation 以解析式 \\(\\frac{2N}{N-1}\\operatorname{Var}(Y_t)|E_t|\\) 計算；先分別加總能量與其期望，再求比值，並非平均每張快照的 \\(H\\)。零變異或沒有邊時需檢查定義適用性，不能把不可量測值當成證據。','\\(H\\) uses observed graphs and aligned targets, with each snapshot’s permutation expectation computed analytically as \\(\\frac{2N}{N-1}\\operatorname{Var}(Y_t)|E_t|\\). Energies and expectations are summed separately before taking their ratio; snapshot \\(H\\) values are not simply averaged. Degenerate variance or edgeless graphs require checking validity rather than treating an undefined quantity as evidence.')}
      ${equation(String.raw`H_i^{\mathrm{att}}=-\frac{\sum_{j\in\widetilde{\mathcal N}(i)}\alpha_{ij}\log\alpha_{ij}}{\log|\widetilde{\mathcal N}(i)|},\quad R=\frac1N\sum_i(1-H_i^{\mathrm{att}})`,'(S8)')}
      ${p('熵帶負號。均勻權重對應 \\(R=0\\)；少數來源主導時 \\(R\\) 接近 \\(1\\)。\\(\\widehat R\\) 由固定 source-feature edge-gating probe 在訓練標籤上 fit 80 epochs，並記錄 \\(\\widehat R-R_{\\mathrm{true}}\\)；它不是 GAT 權重，也不是保證無偏的真值估計。僅有自身來源時定義 \\(R=0\\)。','Entropy includes the minus sign. Uniform weights give \\(R=0\\), while concentration approaches \\(1\\). \\(\\widehat R\\) comes from an 80-epoch source-feature edge-gating probe fitted to training labels, with \\(\\widehat R-R_{\\mathrm{true}}\\) recorded. It is neither GAT attention nor a guaranteed unbiased estimate. A self-only neighborhood is assigned \\(R=0\\).')}
      ${equation(String.raw`D_{\mathrm{struct}}=1-\frac1{K-1}\sum_{t=1}^{K-1}\frac{|E_t\cap E_{t+1}|}{|E_t\cup E_{t+1}|}`,'(S9)')}
      ${p('\\(K\\) 是對齊的訓練輸入快照數：一步任務 \\(K=23\\)、denoising \\(K=24\\)。\\(D_{\\mathrm{struct}}\\) 的 Jaccard 尺度與 \\(\\delta\\)／\\(\\eta\\) 的邊替換比例不同。\\(D_{\\mathrm{pred}}\\) 是固定 probe 的分數差，若觀測圖只有雜訊，可能小於零，應檢查「沒有正的動態收益」，而非強迫它等於零。所有 probe 診斷依賴所選估計器，不能稱為完全 model-independent。','\\(K\\) counts aligned training-input snapshots: 23 for the next-step task and 24 for denoising. Jaccard \\(D_{\\mathrm{struct}}\\) differs in scale from \\(\\delta\\)/\\(\\eta\\) edge-turnover controls. \\(D_{\\mathrm{pred}}\\) is a fixed-probe score gap and can be negative for noisy observations; the relevant claim is no positive dynamic gain, not necessarily exact zero. Probe diagnostics depend on their estimators and are not wholly model-independent.')}
    </details>


    ${heading('5.4','五模型如何公平比較？','How are the five models compared fairly?','sim-evaluation')}
    ${p('五模型使用同一資料種子與時間切分，前處理只用訓練統計；每模型以相同預算比較四個候選，只用 validation loss 選候選與 checkpoint。設定固定後才計算 test 分數。','The five models share dataset seeds and chronological splits, with training-only preprocessing. Each receives the same budget of four candidates; validation loss selects the candidate and checkpoint before test scoring.')}
    <details class="results-details"><summary>${t('展開完整公平性程序與超參數','Expand the complete comparison protocol and hyperparameters')}</summary>
    ${table([t('項目','Item'),t('共同程序','Shared protocol')],[
      [t('資料與時間邊界','Data and time boundaries'),t('同 seed、同資料、同切分。一步目標時間 2–24 訓練、25–32 驗證、33–40 測試；訓練輸入起點為 \\(t\\in\\{1,\\ldots,23\\}\\)。','Same seed, dataset, and split. One-step target times 2–24 train, 25–32 validation, 33–40 test; training input origins are \\(t\\in\\{1,\\ldots,23\\}\\).')],
      [t('架構與資訊','Architecture and information'),t('nonlinear node encoder、單層 aggregation、自身分支與 nonlinear readout；GAT 單頭。MLP 沒有鄰居 aggregation。','Nonlinear node encoder, one aggregation layer, self branch, and nonlinear readout; single-head GAT. MLP has no neighbor aggregation.')],
      [t('損失與調參','Loss and tuning'),t('Regression：MSE；classification：BCE。\\(\\text{Width}\\in\\{32,64\\}\\;\\times\\;\\text{learning rate}\\in\\{0.001,0.003\\}\\)，每模型四候選。','MSE for regression; BCE for classification. \\(\\text{Width}\\in\\{32,64\\}\\;\\times\\;\\text{learning rate}\\in\\{0.001,0.003\\}\\): four candidates per model.')],
      [t('優化與停止','Optimization and stopping'),t('AdamW，\\(\\text{weight decay}=10^{-4}\\)；最多 200 epochs、\\(\\mathrm{patience}=25\\)、\\(\\mathrm{min\\_delta}=10^{-6}\\)。','AdamW, \\(\\text{weight decay}=10^{-4}\\); 200-epoch cap, \\(\\mathrm{patience}=25\\), \\(\\mathrm{min\\_delta}=10^{-6}\\).')],
      [t('模型選擇','Model selection'),t('只用 validation loss 選候選與 checkpoint；固定後才計算 test 分數。','Select candidates/checkpoints using validation loss only, then score the test set.')],
      [t('前處理與容量','Preprocessing and capacity'),t('標準化只用 training statistics；調參預算與容量範圍相同，但 parameter count 不強制相等。','Training-only standardization and equal search/capacity ranges; parameter counts need not be identical.')]
    ])}
    </details>
    <h3>${t('預測分數與配對差值','Prediction scores and paired differences')}</h3>
    ${equation(String.raw`R^2=1-\frac{\sum(Y-\widehat Y)^2}{\sum(Y-\overline Y_{\mathrm{test}})^2},\qquad \Delta_s=\mathrm{Score}_A(D_s)-\mathrm{Score}_B(D_s)`,'(S10)')}
    ${p('主回歸分數為測試 \\(R^2\\)，越高越好，也可為負；另報 RMSE 與 MAE。合成分類穩健性由不含反應雜訊的 latent 連續目標 \\(Y^*\\)，以其 training 第 80 百分位切成二元標籤，主要分數 AP，另報 AUROC、Brier 與 log loss。分類與回歸分開報告，不把不同任務的分數合併平均。AP 使用 recall 增量加權 precision，與梯形 PR-AUC 不完全相同。','Primary regression uses test \\(R^2\\), where higher is better and negative values are possible, with RMSE and MAE reported separately. Synthetic classification robustness thresholds the noiseless latent continuous target \\(Y^*\\) at its training 80th percentile and primarily reports AP, plus AUROC, Brier, and log loss. Classification and regression are reported separately rather than averaging scores across tasks. AP weights precision by recall increments and differs from trapezoidal PR-AUC.')}
    ${p('同一資料種子上的模型差形成 \\(\\Delta_s\\)；以完整資料集種子為單位做 2,000 次 paired bootstrap，報平均差、95% CI 與五個種子中 \\(\\Delta_s>0\\) 的比例。不能把同一圖中的節點或時間當成獨立重複；只有五個 seeds 的不確定性仍有限，正差值比例也不是 \\(p\\)-value 或後驗機率。','Model gaps \\(\\Delta_s\\) are paired within each dataset seed. Two thousand paired bootstrap resamples of whole dataset seeds give the mean gap, 95% CI, and fraction of the five seeds with positive gaps. Nodes and times within a graph are not independent replicates. Five seeds provide limited uncertainty resolution; the positive fraction is not a \\(p\\)-value or posterior probability.')}
    ${table([t('對照','Contrast'),t('回答的問題','Question answered')],[
      ['Each \\(\\operatorname{GNN}-\\operatorname{MLP}\\)',t('圖與聚合是否提供收益？逐模型報告，不用測試集最佳 GNN 的最大值。','Does graph aggregation help? Report each model; do not select the best test-set GNN.')],
      ['\\(\\operatorname{GAT}_{S}-\\operatorname{GCN}_{S}\\; /\\; \\operatorname{GAT}_{D}-\\operatorname{GCN}_{D}\\)',t('同一 topology policy 下的 architecture 差異。','Architecture differences under the same topology policy.')],
      ['\\(\\operatorname{GCN}_{D}-\\operatorname{GCN}_{S}\\; /\\; \\operatorname{GAT}_{D}-\\operatorname{GAT}_{S}\\)',t('同一 architecture 下的 topology policy 差異。','Topology-policy differences within an architecture.')],
      ['\\(I=(\\operatorname{GAT}_{D}-\\operatorname{GAT}_{S})-(\\operatorname{GCN}_{D}-\\operatorname{GCN}_{S})\\)',t('動態拓樸的收益是否依 architecture 而改變？','Does dynamic-topology benefit interact with architecture?')]
    ])}


    <p><a href="#results">${t('查看已完成的模擬結果','Explore the completed simulation results')} →</a></p>
      `,
      results:()=>`
    ${chapterHeader("06","SIMULATION RESULTS","模擬結果","Simulation Results","依研究問題選擇結果分頁；每次聚焦一組圖表、解讀與配對證據。","Choose an evidence tab to focus on one set of figures, interpretation, and paired comparisons.",t)}
    ${resultsMarkup(t)}
    <p><a href="#design">${t('回看實驗與評估定義','Review experimental and evaluation definitions')}</a> · <a href="#discussion">${t('閱讀研究討論與適用限制','Read discussion and limits')} →</a></p>
      `,
      discussion:()=>`
    ${chapterHeader("07","DISCUSSION & CONCLUSIONS","討論與結論","Discussion & Conclusions","整合已完成的機制證據，區分穩定觀察、未成立的假說與推論限制。","Synthesize completed mechanism evidence while separating supported observations, unsupported hypotheses, and inferential limits.",t)}
    ${chapterOutline([["7.1","sim-conclusions","研究結論","Conclusions"],["7.2","discussion-limits","適用限制","Limits"]],t)}
    ${heading('7.1','研究結論與適用限制','Conclusions and scope','sim-conclusions')}
    ${p('已完成的模擬支持一個有條件的結論：圖的增量資訊、鄰居重要性與拓樸變動的預測價值，可以解釋部分模型相對行為；模型名稱本身不足以決定優劣。判讀依據是同一資料種子上的配對分數差、區間與機制反應，而非從測試集挑出一個總冠軍。','The completed simulations support a conditional conclusion: incremental graph information, neighbor relevance, and the predictive value of topology change explain part of relative model behavior. A model name alone does not determine its advantage. Evidence comes from paired score differences, intervals, and mechanism responses within the same dataset seeds, rather than selecting an overall test-set winner.')}
    ${table([t('機制問題','Mechanism question'),t('已完成結果支持的觀察','Observation supported by completed results'),t('解讀界線','Interpretive boundary')],[
      [t('圖是否增加資訊？','Does the graph add information?'),t('增加 \\(\\lambda\\) 時，GCN 相對 MLP 的收益隨圖訊號增加；\\(\\lambda=0\\) 時 MLP 仍具競爭力。','Increasing \\(\\lambda\\) increases GCN gains over MLP alongside graph signal; MLP remains competitive at \\(\\lambda=0\\).'),t('收益依賴此生成器的鄰居訊號，不代表任意圖都應加入模型。','Gains depend on the neighbor signal in this generator; they do not justify adding an arbitrary graph.')],
      [t('何時需要選擇性聚合？','When does selective aggregation help?'),t('增加 \\(\\kappa\\) 時，GAT 相對 GCN 的配對收益增加；均勻鄰居下差異很小。','Increasing \\(\\kappa\\) increases the paired GAT gain over GCN; differences are small under uniform neighbors.'),t('GCN 的均勻聚合對齊依賴 regular graph；圖類型與正規化也會影響架構差。','Uniform-aggregation alignment for GCN depends on regular graphs; graph family and normalization also affect architecture gaps.')],
      [t('平滑是否有助去雜訊？','Does smoothness help denoising?'),t('獨立 \\(\\tau\\) 實驗中，圖聚合相對 MLP 的收益增加。','Graph-aggregation gains over MLP increase in the isolated \\(\\tau\\) experiment.'),t('由帶雜訊目標估計的 \\(H\\) 並非單調，不能把控制量與量測指標視為同一量。','\\(H\\) estimated from noisy targets is not monotone; a control and a measured diagnostic are distinct quantities.')],
      [t('何時值得更新拓樸？','When is topology updating useful?'),t('乾淨且具預測價值的真實變動有利於 Dynamic；只有觀測圖雜訊時，Static 相對更有利。','Clean predictive changes in true topology favor Dynamic; observation noise alone favors Static.'),t('動態收益不隨 \\(\\delta\\) 嚴格單調；\\(D_{\\mathrm{struct}}\\) 不能取代 \\(D_{\\mathrm{pred}}\\)。','Dynamic gain is not strictly monotone in \\(\\delta\\); \\(D_{\\mathrm{struct}}\\) cannot replace \\(D_{\\mathrm{pred}}\\).')]
    ])}
    ${p('多軸相圖將這些觀察放在同一診斷空間，呈現連續的模型表現差。其相關性來自機制同時存在的條件，不能解讀為控制其他因素後的因果效應；也未建立保證選中最佳模型的決策規則。尤其 \\(\\widehat R\\) 的排序能力與數值校準不同，靜態與動態架構的關聯方向也可能不同，必須連同圖品質與其他診斷共同解讀。','The multi-axis maps place these observations in a common diagnostic space and display continuous performance differences. Associations under coexisting mechanisms are not causal effects adjusted for other factors, and they do not establish a rule guaranteed to select the best model. Rank tracking by \\(\\widehat R\\) is distinct from numerical calibration; associations may differ between static and dynamic architectures and require joint interpretation with graph quality and other diagnostics.')}
    ${heading('7.2','適用限制與版本界線','Limits and version boundaries','discussion-limits')}
    ${table([t('限制','Limitation'),t('解讀方式','Interpretation')],[
      [t('指定的生成器與架構','Specific generator and architectures'),t('結論適用於此訊號機制、snapshot 模型與預算，不能視為所有 GNN 的定理。','Conclusions concern these mechanisms, snapshot architectures, and budgets, not all GNNs.')],
      [t('Probe 的能力','Probe capacity'),t('低 \\(S_G\\)／\\(\\widehat R\\) 不一定表示沒有圖資訊；需連同打亂對照與 \\(\\widehat R-R_{\\mathrm{true}}\\) 評估。','Low \\(S_G\\)/\\(\\widehat R\\) need not mean no graph information; inspect shuffled controls and \\(\\widehat R-R_{\\mathrm{true}}\\).')],
      [t('統計不確定性','Statistical uncertainty'),t('只有五個獨立種子；逐條件區間沒有多重比較校正，不能以正差值比例代替假說檢定。','There are only five independent seeds and no multiple-comparison adjustment for condition-level intervals; positive-gap fractions do not replace hypothesis tests.')],
      [t('有限的訓練預算','Finite training budget'),t('達 epochs 上限時仍有訓練預算限制；六個機制情境的預算檢查不構成所有實驗已收斂的證明。','Epoch-cap flags remain material limitations; budget checks on six mechanism regimes do not prove convergence across all experiments.')],
      [t('任務與時間對齊','Task and temporal alignment'),t('主一步回歸、同時點去雜訊與合成二元分類使用不同標籤及分數，需分開討論。','Primary one-step regression, same-time denoising, and synthetic binary classification use different labels and scores and are interpreted separately.')]
    ])}
    ${note(t('本研究的成果','Study contribution'),'本研究完成合成資料生成、五模型配對比較、機制反應曲線、診斷相圖與穩健性分析，並提供逐種子分數及來源校驗。結論是指定模擬機制範圍內可重算的經驗證據；模型的相對優勢必須與資料機制、診斷估計誤差及訓練條件一起說明。','The study provides synthetic data generators, paired five-model comparisons, mechanism response curves, diagnostic maps, robustness analyses, and seed-level scores with provenance checksums. Its conclusions are reproducible empirical evidence within the specified simulated mechanisms. Relative model advantages are interpreted jointly with data mechanisms, diagnostic-estimation error, and training conditions.')}
    <details class="sim-v2-legacy"><summary>${t('第一版結果的版本界線','Boundary with archived V1 results')}</summary>${p('第一版 945 次評估使用不同節點數、歷史輸入、分類任務與建圖程序，原始下載資料仍保留。本章採第二版，兩版分數不合併平均，也不將第一版的選模樹表現當成第二版結論。','V1’s 945 evaluations used different node counts, historical inputs, classification targets, and construction procedures. Its downloadable artifacts are retained. This chapter uses V2; scores are not pooled, and V1’s selection-tree performance is not a V2 conclusion.')}<p><a href="data/simulation/provenance.json" target="_blank" rel="noopener">${t('第一版來源与校驗紀錄','V1 provenance and checksums')}</a></p></details>

    <p><a href="#results">${t('核對結果圖表與配對區間','Inspect the figures and paired intervals')}</a> · <a href="#design">${t('回看實驗設計','Review experimental design')}</a></p>
      `
    };
    return views[view]();
  }
  for(const chapter of ['generation','design','results','discussion']) window[chapter+'Chapter']=(...args)=>chapterView(chapter,...args);
}());
