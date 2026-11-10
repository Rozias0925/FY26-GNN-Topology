# Static and Dynamic Graph Construction for GNNs

本專題以可控制模擬研究 MLP、GCN、GAT 與靜態／動態圖的相對優勢，從生成機制、可觀測診斷與配對表現差建立解釋。現行研究範圍見 [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md)。

## 七個閱讀章節

| 章節 | 閱讀內容 |
|---|---|
| 01 Overview／研究概述 | 摘要、三個研究問題、閱讀指南與成果邊界。 |
| 02 Graphs & Models／圖與模型 | 圖與五模型的定義、聚合公式及預測架構；術語及分類細節可展開。 |
| 03 Interactive Example／互動算例 | 四節點的獨立教學頁，逐步調整特徵、時間與模型。 |
| 04 Data Generation／資料生成 | 特徵、目標、平滑訊號、真實／觀測圖及模型輸入。 |
| 05 Experiments & Evaluation／實驗與評估 | 機制情境、Stage A–D、診斷定義、公平比較與配對推論。 |
| 06 Simulation Results／模擬結果 | 七個結果子分頁，一次顯示一組證據；完整表格與下載可展開。 |
| 07 Discussion & Conclusions／討論與結論 | 跨實驗綜合解讀、假說支持程度與適用限制。 |

網站使用深綠背景與月光色閱讀面板，支援繁體中文／English；保留 Rozias logo、Researcher Profile 與本機 LaTeX 公式排版。語言偏好只儲存在瀏覽器本機。

2026-11-10 已依研究範圍調整移除資料／背景與水庫案例章節，並依閱讀目的拆成七章。先前的實測資料、分析程式及既有成果保留在本機，不是完成目前專題所需的後續工作。舊網頁內容保留在 [archive/website-before-simulation-only-20261110](archive/website-before-simulation-only-20261110/README.md)，不隨 `dist/` 發布。

## 完成的模擬研究

正式比較五模型：MLP、Static-GCN、Static-GAT、Dynamic-GCN、Dynamic-GAT。主實驗採一步連續回歸，平滑度實驗採同時點去噪；二元分類與 GATv2 分別列於穩健性實驗。合成時間步不自動代表一小時。

完整批次於 2026-10-05 23:41（台北時間）完成：284 個工作、7,240 次模型評估；另列原 Stage A 的 150 次評估。各組皆使用配對資料與既定時間切分。結果按主要 200 epochs 與 400 epochs 敏感度分開展示，不依測試分數挑選訓練預算。

網站結果分成七區：Stage A 機制檢查、五條反應曲線、三張相圖、200 個多軸條件、樣本數、穩健性、訓練預算敏感度。提供 14 張科學圖與 22 個下載項目，以及中英文解讀。研究不要求任一模型必須勝出，也不把診斷相圖當成通用的模型推薦規則。

模型優勢須連同非單調反應、診斷估計誤差、五個種子的不確定性及收斂限制閱讀。相圖說明目前生成機制下的相對行為，無法直接保證任意新資料上的表現。

## 開啟與發布

`dist/` 是直接編輯的靜態網站來源，無需 build 或 API key。可直接開啟 `dist/index.html`，或在專案根目錄啟動本機預覽：

```powershell
& algorithm/.venv/Scripts/python.exe -m http.server 4173 --bind 127.0.0.1 --directory dist
```

瀏覽 [本機預覽](http://127.0.0.1:4173/)。既有預覽分頁請重新整理。

程式碼儲存庫：[Rozias0925/FY26-GNN-Topology](https://github.com/Rozias0925/FY26-GNN-Topology)。GitHub Pages 工作流程為 `.github/workflows/pages.yml`，只發布 `dist/`。Repository 的 Pages Source 使用 GitHub Actions；推送 `main`／`master` 或手動執行 **Deploy research website** 可發布。實際部署狀態請以儲存庫的 Actions 執行結果為準。

## 網頁與數據檔案

```text
dist/
  index.html                       分組的七章導覽、語言切換與 metadata
  content.js                       Overview／Models／Walkthrough 雙語正文與章節分派
  report.js                        七章路由、雙語與互動教學
  simulation-v2.js                  Generation／Design／Results／Discussion 與結果分頁
  report.css / shell.css            深綠與月光色版面
  simulation-v2.css / math-layout.css / chapter-layout.css
  math.js                          未訓練教學算例的純數學計算
  data/mechanism-results-web.js     瀏覽器載入的完整第二版結果
  data/mechanism-results-v2.json    原始完整匯出
  data/mechanism-discussion-v2.js   七個結果區的中英文解讀
  data/mechanisms-v2/               CSV、原圖及來源校驗碼
  assets/figures-latex/             14 張 LaTeX 標籤展示圖及 manifest
  assets/vendor/katex/              本機渲染、字型與授權
  assets/brand/                     Rozias 標誌
algorithm/src/gnn_mechanisms/        第二版生成器、模型、診斷、訓練與報表
algorithm/tests/                    方法、時間切分與完整性測試
scripts/                           匯出與網站數據驗證工具
archive/                           不發布的歷史網頁內容
```

五模型的四節點教學算例使用指定算例參數計算係數，示範「輸入、權重、加總、ReLU」。輸出是節點表示；不當成訓練成績。所有正文公式、數值表與圖上數學標籤使用本機 KaTeX；章節、語言及控制項更新後重新排版。舊 `#simulation` 網址轉至「實驗與評估」；已移除的案例網址轉回研究概述。

## 驗證與重建

演算法環境位於 `algorithm/.venv`。完整程式測試、原始資料與既有成果檢查已於 2026-11-03 完成；101 項演算法測試與 12 項匯出器測試通過。

本次網站範圍修改不重新訓練模型、不重算配對區間、不改任何研究數值。僅需驗證七章雙語路由、結果子分頁、公式、互動算例與來源完整性。完整模擬成果可用唯讀入口重新核對：

```powershell
& algorithm/.venv/Scripts/python.exe scripts/export_mechanisms.py --check-only
& algorithm/.venv/Scripts/python.exe scripts/render_latex_figures.py --plan-only
```

`--check-only` 檢查完整批次、來源、配對種子、CSV／JSON 與圖表覆蓋。`--plan-only` 驗證 14 張展示圖的來源計畫，不載入數值套件或重繪。

`mechanism-results-web.js` 由 `scripts/prepare_website_results.cjs` 根據原 JSON 產生；原始 `mechanism-results-v2.js` 因本機讀取權限問題保留原檔，主頁使用等值的 browser bundle。該工具會驗證 35 個來源檔案與 22 個下載，執行時會寫入 bundle；本次只核對既有內容，沒有重建它。

14 張 LaTeX 展示圖由 `scripts/render_latex_figures.py` 讀取既有 CSV 產生；`presentation_manifest.json` 保留原圖、來源、逐點資料與展示圖 SHA-256。原始結果與展示版本分開保存。

第一版 945 次評估與第二版使用不同生成器、輸入及任務，保留封存資料但不混合計分。三個 condition hashes 曾被 CSV 型別推斷改寫；修復記錄與 7,240 份原始分數逐一核對保存在正式成果的 `report_repair.json`，未重訓權重。既有 repair 腳本不需為本次網站修改重跑。

KaTeX 原始授權保留於 `dist/assets/vendor/katex/LICENSE`。Rozias logo 延用既有個人標誌。
