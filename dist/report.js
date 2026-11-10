(function () {
  'use strict';
  const chapters = ['overview','models','walkthrough','generation','design','results','discussion'];
  const titles = {
    zh:['研究概述','圖與模型','互動算例','資料生成','實驗與評估','模擬結果','討論與結論'],
    en:['Overview','Graphs & Models','Interactive Example','Data Generation','Experiments & Evaluation','Simulation Results','Discussion & Conclusions']
  };
  const subtitles = ['Research questions','Prediction architectures','Step-by-step calculation','Synthetic mechanisms','Protocol and diagnostics','Paired evidence','Scope and limitations'];
  const aliases = {simulation:'design',data:'overview',cases:'overview'};
  const resolveChapter = hash => chapters.includes(hash) ? hash : Object.hasOwn(aliases,hash) ? aliases[hash] : 'overview';
  let language = 'zh';
  try { if (localStorage.getItem('graph-report-language') === 'en') language = 'en'; } catch (_) { /* The report also works with storage blocked. */ }
  let chapter = resolveChapter(location.hash.slice(1));
  if (location.hash && location.hash.slice(1)!==chapter) history.replaceState(null,'',`#${chapter}`);
  const state = {model:'Static GCN',time:0,stage:0,features:[2,1,4,3],featureNode:0};
  const byId = id => document.getElementById(id);
  const t = (zh,en) => language === 'en' ? en : zh;
  const svgNS = 'http://www.w3.org/2000/svg';
  const fixed = value => value.toFixed(3);
  const inline = tex => `\\(${tex}\\)`;
  function el(tag, attrs={}, text) { const element=document.createElementNS(svgNS,tag); for(const [key,value] of Object.entries(attrs)) element.setAttribute(key,String(value)); if(text!==undefined) element.textContent=text; return element; }
  function pressed(selector, attribute, value) {document.querySelectorAll(selector).forEach(button=>button.setAttribute('aria-pressed',String(button.getAttribute(attribute)===String(value))));}
  function math(container) { if(window.renderMathInElement) window.renderMathInElement(container,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],throwOnError:false,strict:'warn',trust:false}); }
  window.renderReportMath = math;
  function graphMath(cx, cy, tex, color, size=16, width=140) {
    const foreign = el('foreignObject',{x:cx-width/2,y:cy-22,width,height:44});
    const label = document.createElementNS('http://www.w3.org/1999/xhtml','div');
    label.className='graph-math-label';
    label.style.color=color;
    label.style.fontSize=`${size}px`;
    if(window.katex) window.katex.render(tex,label,{throwOnError:false,trust:false,strict:'warn'});
    else label.textContent=tex;
    foreign.appendChild(label);
    return foreign;
  }
  function closeMenu(){byId('chapter-sidebar').classList.remove('open');byId('menu-toggle').setAttribute('aria-expanded','false');}
  document.querySelector('.skip-link').addEventListener('click',event=>{
    event.preventDefault();closeMenu();byId('main').focus({preventScroll:true});
    byId('main').scrollIntoView({block:'start',behavior:'instant'});
  });
  byId('menu-toggle').addEventListener('click',()=>{const open=byId('menu-toggle').getAttribute('aria-expanded')!=='true';byId('menu-toggle').setAttribute('aria-expanded',String(open));byId('chapter-sidebar').classList.toggle('open',open);});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&byId('menu-toggle').getAttribute('aria-expanded')==='true'){closeMenu();byId('menu-toggle').focus();}});
  document.querySelectorAll('[data-language]').forEach(button=>button.addEventListener('click',()=>{
    language=button.dataset.language;
    try{localStorage.setItem('graph-report-language',language);}catch(_){}
    const y=window.scrollY;renderChapter(false);window.scrollTo({top:y,behavior:'instant'});
  }));
  document.querySelectorAll('.chapter-nav a').forEach(link=>link.addEventListener('click',()=>{closeMenu();if(link.hash===location.hash)window.scrollTo({top:0,behavior:'smooth'});}));
  window.addEventListener('hashchange',()=>{
    const requested=location.hash.slice(1);
    chapter=resolveChapter(requested);
    if(requested!==chapter)history.replaceState(null,'',`#${chapter}`);
    closeMenu();renderChapter(true);
  });

  function renderChrome(){
    document.documentElement.lang=language==='en'?'en':'zh-Hant';
    document.title=`${titles[language][chapters.indexOf(chapter)]} | Static and Dynamic Graph Construction for GNNs`;
    byId('site-description').content=t('數學專題研究：以可控制模擬、配對模型比較與診斷相圖，研究 MLP、GCN、GAT 及靜態／動態圖的相對優勢。','A mathematics research project examining the relative advantages of MLP, GCN, GAT, and static/dynamic graphs through controlled simulation, paired comparisons, and diagnostic maps.');
    document.querySelectorAll('[data-zh][data-en]').forEach(element=>element.textContent=element.dataset[language]);
    pressed('[data-language]','data-language',language);
    document.querySelectorAll('.chapter-nav a').forEach((link,i)=>{link.querySelector('b').textContent=titles[language][i];link.querySelector('small').textContent=language==='zh'?titles.en[i]:subtitles[i];if(chapters[i]===chapter)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');});
    byId('chapter-nav').setAttribute('aria-label',t('七個研究章節','Seven research chapters'));
    byId('language-controls').setAttribute('aria-label',t('語言切換','Language'));
    const index=chapters.indexOf(chapter);
    byId('chapter-pagination').replaceChildren();
    for(const delta of [-1,1]){
      const wrapper=document.createElement('span');
      if(index+delta>=0&&index+delta<chapters.length){const link=document.createElement('a');link.href=`#${chapters[index+delta]}`;link.textContent=delta<0?`← ${titles[language][index-1]}`:`${titles[language][index+1]} →`;wrapper.appendChild(link);}
      byId('chapter-pagination').appendChild(wrapper);
    }
  }
  function renderChapter(scroll){
    renderChrome();
    byId('report').dataset.chapter=chapter;
    byId('report').innerHTML=window.reportChapter(chapter,language);
    if(chapter==='walkthrough')initLesson();
    if(['generation','design','results','discussion'].includes(chapter))window.MechanismResults.mount(t,chapter);
    math(byId('report'));
    if(scroll){window.scrollTo({top:0,behavior:'instant'});const title=byId('report').querySelector('h1');title.setAttribute('tabindex','-1');title.focus({preventScroll:true});}
  }

  function initLesson(){
    byId('lesson-model').value=state.model;
    byId('feature-node').value=String(state.featureNode);
    byId('lesson-model').addEventListener('change',event=>{state.model=event.target.value;state.stage=0;renderLesson();});
    byId('feature-node').addEventListener('change',event=>{state.featureNode=Number(event.target.value);renderLesson();});
    byId('feature-value').addEventListener('input',event=>{state.features[state.featureNode]=Number(event.target.value);renderLesson();});
    document.querySelectorAll('[data-lesson-time]').forEach(button=>button.addEventListener('click',()=>{state.time=Number(button.dataset.lessonTime);renderLesson();}));
    byId('lesson-previous').addEventListener('click',()=>{state.stage=Math.max(0,state.stage-1);renderLesson();});
    byId('lesson-next').addEventListener('click',()=>{state.stage=Math.min(3,state.stage+1);renderLesson();});
    byId('lesson-reset').addEventListener('click',()=>{Object.assign(state,{model:'Static GCN',time:0,stage:0,features:[2,1,4,3],featureNode:0});byId('lesson-model').value=state.model;byId('feature-node').value='0';renderLesson();});
    renderLesson();
  }
  function renderLesson(){
    const result=GraphLesson.modelExample(state.model,state.time,state.features);
    pressed('[data-lesson-time]','data-lesson-time',state.time);
    byId('feature-value').value=String(state.features[state.featureNode]);
    byId('feature-output').value=String(state.features[state.featureNode]);
    byId('feature-value').setAttribute('aria-valuetext',`${['i','B','C','D'][state.featureNode]}: ${state.features[state.featureNode]}`);
    byId('lesson-previous').disabled=state.stage===0;
    byId('lesson-next').disabled=state.stage===3;
    byId('lesson-step-count').textContent=`${state.stage+1} / 4`;
    const names=t(['輸入特徵','建立權重','加總訊息','非線性更新'],['Input features','Compute weights','Aggregate messages','Nonlinear update']);
    byId('lesson-step-title').textContent=names[state.stage];
    const description=state.stage===0?t(`${inline('h^{(0)}')} 就是初始特徵。先確認目標節點與哪些鄰居相連，再決定哪些資訊能參與計算。`,`${inline('h^{(0)}')} contains the initial features. Adjacency determines which neighbors may contribute to the target.`):
      state.stage===1?(state.model==='MLP'?t(`MLP 只保留目標自身，這個單層算例的 ${inline('W=1')}。`,`MLP retains only the target; ${inline('W=1')} in this one-layer example.`):result.attention?t(`對允許的鄰居與自身，以 ${inline(String.raw`\operatorname{LeakyReLU}(h_j)`)} 計分後做 softmax；權重會隨特徵改變。`,`Score allowed neighbors and self with ${inline(String.raw`\operatorname{LeakyReLU}(h_j)`)}, then apply softmax. Weights depend on features.`):t(`含自迴路後，目標度數為 ${inline('3')}、相連鄰居度數為 ${inline('2')}；自身係數為 ${inline(String.raw`\frac13`)}，鄰居係數為 ${inline(String.raw`\frac1{\sqrt6}`)}。`,`With self-loops, target degree is ${inline('3')} and connected-neighbor degree is ${inline('2')}. The self coefficient is ${inline(String.raw`\frac13`)}; each neighbor coefficient is ${inline(String.raw`\frac1{\sqrt6}`)}.`)):
      state.stage===2?t('將每個允許來源的特徵乘上係數後相加。未連接節點的特徵不會直接進入這個單層更新。','Multiply each allowed source feature by its coefficient and sum. Disconnected nodes do not directly enter this one-layer update.'):
      t(`${inline(String.raw`\operatorname{ReLU}(z)=\max(0,z)`)}。輸出是更新後的節點表示；還需要預測頭及訓練，才能用於實際預測。`,`${inline(String.raw`\operatorname{ReLU}(z)=\max(0,z)`)}. This is an updated node representation; a trained prediction head is still needed for forecasting.`);
    byId('lesson-step-description').textContent=description;
    const labels=['i',String.raw`\mathrm B`,String.raw`\mathrm C`,String.raw`\mathrm D`];
    byId('lesson-weights').innerHTML=`<div class="table-wrap"><table><thead><tr><th>${t('來源','Source')}</th><th>${inline('h_j')}</th><th>${t('係數','Weight')}</th><th>${t('貢獻','Contribution')}</th></tr></thead><tbody>${result.terms.map(term=>`<tr><th>${inline(labels[term.node])}</th><td>${inline(String(term.feature))}</td><td>${state.stage>=1?inline(fixed(term.weight)):'—'}</td><td>${state.stage>=2?inline(fixed(term.contribution)):'—'}</td></tr>`).join('')}</tbody></table></div>`;
    byId('lesson-result').className='lesson-result';
    byId('lesson-result').innerHTML=state.stage<2?`<p>${t('按「下一步」查看計算。','Select Next to inspect the calculation.')}</p>`:state.stage===2?`<strong>${inline(`z=${result.terms.map(term=>term.contribution<0?`(${fixed(term.contribution)})`:fixed(term.contribution)).join('+')}=${fixed(result.aggregate)}`)}</strong><p>${t('顯示值四捨五入至三位小數；加總使用未捨入值。','Displayed values are rounded to three decimals; totals use full precision.')}</p>`:`<strong>${inline(String.raw`h'_i=\max(0,${fixed(result.aggregate)})=${fixed(result.output)}`)}</strong><p>${t('教學算例，不是事件機率或模型成績。','Teaching calculation, not an event probability or benchmark score.')}</p>`;
    const svg=byId('lesson-graph');svg.replaceChildren();
    svg.appendChild(el('title',{},`${state.model}; t${state.time}; ${t('目前加總','Current aggregate')} ${fixed(result.aggregate)}`));
    const coordinates=[[215,215],[65,70],[215,55],[365,70]];
    [1,2,3].forEach(node=>{
      const active=result.neighbors.includes(node);const [x1,y1]=coordinates[node];
      const term=result.terms.find(item=>item.node===node);
      svg.appendChild(el('line',{x1,y1:y1+24,x2:215,y2:190,stroke:active?'#486b40':'#bfc8b1','stroke-width':active&&state.stage>=1?(1+term.weight*5):1.5,'stroke-dasharray':active?'':'4 6',opacity:active?1:.6}));
      if(active&&state.stage>=1)svg.appendChild(graphMath((x1+215)/2+(node===2?29:0),(y1+215)/2-4,fixed(term.weight),'#745f2f',15,76));
    });
    if(state.model!=='MLP')svg.appendChild(el('path',{d:'M192 226 C153 285 276 285 238 226',fill:'none',stroke:'#708655','stroke-width':1.5}));
    coordinates.forEach(([cx,cy],index)=>{
      const active=result.active.includes(index);svg.appendChild(el('circle',{cx,cy,r:25,fill:index===0?'#325a40':active?'#dce5c9':'#ebeee2',stroke:active?'#66854f':'#b8c4a7','stroke-width':1.5}));
      svg.appendChild(graphMath(cx,cy,labels[index],index===0?'#f4f1df':'#34502e',18,44));
      svg.appendChild(graphMath(cx,cy-43,`h_{${labels[index]}}^{(0)}=${state.features[index]}`,'#435c38',16,120));
    });
    byId('lesson-graph-caption').textContent=state.model==='MLP'?t(`MLP 不讀取 ${inline(String.raw`\mathrm B,\mathrm C,\mathrm D`)}。虛線只顯示候選位置，不參與計算。`,`MLP does not read ${inline(String.raw`\mathrm B,\mathrm C,\mathrm D`)}. Dashed lines show candidate relations and do not participate.`):
      result.dynamic?t(`Dynamic：${inline('t_0')} 使用 ${inline(String.raw`\mathrm B,\mathrm C`)}；${inline('t_1')} 使用 ${inline(String.raw`\mathrm B,\mathrm D`)}。虛線表示未使用的關係。`,`Dynamic: ${inline(String.raw`\mathrm B,\mathrm C`)} at ${inline('t_0')}; ${inline(String.raw`\mathrm B,\mathrm D`)} at ${inline('t_1')}. Dashed relations are unused.`):
      t(`Static：兩個時間點都使用 ${inline(String.raw`\mathrm B,\mathrm C`)}。修改 ${inline(String.raw`\mathrm C`)} 的特徵，可觀察 GAT 的權重如何改變。`,`Static: ${inline(String.raw`\mathrm B,\mathrm C`)} are used at both times. Change ${inline(String.raw`\mathrm C`)}’s feature to inspect feature-dependent GAT weights.`);
    for(const id of ['lesson-step-description','lesson-weights','lesson-result','lesson-graph-caption']) math(byId(id));
  }

  renderChapter(false);
})();
