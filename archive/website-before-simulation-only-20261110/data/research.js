/* Conceptual teaching fixtures. V2 research outputs are separate verified artifacts. */
window.RESEARCH_DATA = {
  schemaVersion: 1,
  status: 'pending',
  models: ['MLP', 'Static GCN', 'Static GAT', 'Dynamic GCN', 'Dynamic GAT'],
  horizons: [1, 3, 6],
  results: { simulation: null, selectionMap: null, indicators: null, hydrology: null, events: null, metrics: null },
  evaluation: {
    classification: {
      metrics: ['AUROC', 'Precision', 'Recall', 'F1', 'False alarm rate', 'Event detection rate'],
      note: '分類以時間窗標籤評估；事件偵測率以事件為單位。警報閾值、事件合併與 false alarm rate 的分母需在測試前固定。'
    },
    warning: {
      metrics: ['Warning lead time', 'Event detection rate', 'False alarm rate'],
      note: '預警提前時間以事件起點減去有效預警時間計算，需事先定義事件起點、警報觸發條件與跨窗去重方式。'
    },
    regression: {
      metrics: ['MAE', 'RMSE'],
      note: '回歸評估第 h 小時的濁度預測；量測單位與各時間範圍將在真實資料接入後標明。'
    }
  },
  demo: {
    label: 'Conceptual demonstration — not experimental results',
    defaults: { signal: 60, drift: 35, noise: 15, concentration: 40 },
    nodes: [[82,78],[197,48],[322,76],[449,63],[54,191],[172,154],[296,176],[447,190],[99,316],[217,278],[348,311],[493,299],[271,381],[399,398]],
    edges: [[0,1],[0,4],[0,5],[1,2],[1,5],[2,3],[2,6],[3,7],[4,5],[4,8],[5,6],[5,9],[6,7],[6,9],[6,10],[7,11],[8,9],[9,10],[9,12],[10,11],[10,13],[12,13]],
    alternatives: [[0,6],[1,6],[2,7],[3,6],[4,9],[5,10],[6,11],[7,10],[8,12],[9,13]],
    noiseEdges: [[0,13],[1,11],[3,8],[4,10],[2,12],[7,12]],
    snapshots: ['t₀', 't₁', 't₂', 't₃', 't₄'],
    hydrologyWeights: [[.3,.5,.25,.4,.45],[.5,.4,.6,.55,.6],[.85,.65,.5,.8,.75],[.6,.85,.4,.9,.9],[.4,.5,.35,.6,.7]]
  }
};
