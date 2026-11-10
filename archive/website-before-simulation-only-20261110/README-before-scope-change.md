# Static and Dynamic Graph Construction for GNNs

以數學系學生專題報告的方式，說明五模型選擇問題、數學定義、候選實驗設計與高濁度事件驗證。保留深綠背景與月光色閱讀面板，完整支援繁體中文／English。

## 五個主題

1. **Overview**：研究摘要、三個問題、先模擬後真實資料驗證的流程、分類／回歸目標及進度。
2. **Concepts & Models**：Graph、節點、邊、特徵、adjacency、embedding、Static／Dynamic、MLP／GCN／GAT 定義與公式，以及逐步計算的互動算例。
3. **Simulation**：第二版生成式、五條單軸機制、Stage A／B／C／D、六項診斷、公平比較及配對推論；七個結果區已接入完整數據、圖表、配對區間及中英文解讀。
4. **Data & Context**：高濁度背景、資料盤點、圖建構、外部驗證程序，以及預警可支援的準備工作。
5. **Event Case Studies**：歷史事件預留區、分類與回歸曲線、命中／漏報／誤報、事件配對、有效提前時間、五模型評估。

頂部使用既有專案的 Rozias logo 與 Researcher Profile 連結。語言偏好只儲存在本機瀏覽器；阻擋儲存時仍可使用切換。

## 演算法實作

`algorithm/` 保留第一版資料下載、品質檢查、訓練、事件評估與選模工具；第二版 `src/gnn_mechanisms/` 實作機制生成器、五模型比較、diagnostics、配對推論、完整模擬排程及續跑。第二版正式模擬與網站接入均已完成，診斷相圖仍待真實資料外部驗證。第一版研究規格與資料限制見 [`algorithm/RESEARCH_PROTOCOL.md`](algorithm/RESEARCH_PROTOCOL.md)。

```powershell
& algorithm\.venv\Scripts\python.exe -m unittest discover -s algorithm/tests -v
& algorithm\.venv\Scripts\python.exe algorithm/run_research.py simulate --config algorithm/configs/smoke.json --output algorithm/results/my_smoke
```

執行及續跑方式見 [`algorithm/README.md`](algorithm/README.md)。已下載的濁度資料，在固定門檻及時間切分下缺少驗證／測試正例；目前不能宣稱完成高濁度事件外部驗證。

官方資料下載及分析的新入口見 [`algorithm/OFFICIAL_DATA_WORKFLOW.md`](algorithm/OFFICIAL_DATA_WORKFLOW.md)：`fetch_official.py` 保存官方原始分頁、目錄、OpenAPI 與 SHA-256，`analyze_official.py` 輸出品質盤點、hourly panel、1／3／6h labels、事件清單、training-only diagnostics、baseline 與科學圖表。原始資料與產生的分析輸出保留在本機，不隨 GitHub Pages 一併發布。

真實資料五模型的探索性回歸比較使用 [`algorithm/REAL_BENCHMARK_RUN.md`](algorithm/REAL_BENCHMARK_RUN.md) 所述的新入口。它先檢查記憶體／CPU，再跑測試及 preflight；以單核心、單執行緒、Idle priority 排隊與續跑。分類 validation 無正例的限制仍保留，不以原本的分類 loss 選擇回歸模型。

## 開啟與發布

直接開啟 `dist/index.html` 即可，不需安裝依賴或使用 API key。所有圖示、算例、字型及 KaTeX 都在本機；文獻連結需要連網。

也可於專案根目錄執行：

```sh
python -m http.server 4173 --bind 127.0.0.1 --directory dist
```

開啟 http://127.0.0.1:4173/ 。既有預覽分頁請重新整理以載入新版。

GitHub Pages：把此專案內容放在 repository 根目錄（包含 `.github`），在 Settings → Pages 的 Source 選擇 GitHub Actions，再推送至 `main`／`master` 或手動執行 **Deploy research website**。工作流程發布 `dist/`；其他分支名稱需修改 `pages.yml`。尚未替你建立 repository 或發布。

[GitHub Pages 官方工作流程說明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

## 檔案結構

```text
dist/
  index.html              頂部身份、語言切換與五章導覽
  report.css              專題報告版面、響應式及列印樣式
  report.js               章節路由、雙語、算例與控制元件
  content.js              Overview／Concepts & Models 雙語正文與共用模板
  research-chapters.js    Data／Cases 雙語正文
  simulation-v2.js        第二版模擬雙語正文、結果區與資料接入
  simulation-v2.css       模擬閱讀順序與結果面板，延用深綠／月光色
  simulation-scenarios.js 第一版情境說明封存（目前主頁不載入）
  simulation-results.js  第一版結果介面封存（目前主頁不載入）
  simulation-results.css 共用表格／details 樣式與第一版介面樣式
  math.js                 獨立的教學算例數學計算
  data/research.js        教學示意圖資料
  data/mechanism-results-v2.js 核對後的完整第二版數據與來源
  data/mechanism-discussion-v2.js 各結果區的中英文研究解讀
  data/mechanisms-v2/     正式匯出後的 CSV、科學圖表與校驗碼
  data/simulation-results.js  第一版結果封存
  data/simulation/        第一版 CSV、設定、規則與校驗碼
  assets/brand/           既有 Rozias 個人標誌
  assets/vendor/katex/   本機公式渲染、字型與原授權
  .nojekyll
algorithm/
  src/gnn_graph_selection/  模擬、資料、五模型、選模、事件與報告
  src/gnn_mechanisms/        第二版機制研究、paired contrasts 與完整批次
  tests/                    核心計算、時間洩漏與資料品質檢查
  run_research.py            模擬、資料稽核及外部驗證命令
  run_batch.py               快照、續跑與背景執行
  README.md                 執行方式與資料限制
.github/workflows/pages.yml
.openai/hosting.json      靜態網站工具的目錄設定；沒有線上 Site 登記
```

`dist/` 是直接編輯、納入 Git 的網站來源。無需 build。

網站五章的中英文數學符號、正文公式、結果表及互動算例統一使用本機 KaTeX。`report.js` 匯出 `renderReportMath`，章節、語言、模型、時間切片、曲線及相圖更新後都重新排版；`math-layout.css` 處理行內公式、圖上標籤與窄螢幕的水平捲動。

14 張科學圖的 LaTeX 展示版放在 `dist/assets/figures-latex/`。`scripts/render_latex_figures.py` 只讀既有配對 CSV，以 Matplotlib mathtext 改寫標籤；不重新計算分數或區間。其 `presentation_manifest.json` 保存來源、原圖、展示圖與逐點資料的 SHA-256，並確認原始成果未變。使用 `algorithm/.venv/Scripts/python.exe scripts/render_latex_figures.py` 重建，或加 `--plan-only` 進行不載入數值套件的盤點。

瀏覽器目前載入 `dist/data/mechanism-results-web.js`，由 `scripts/prepare_website_results.cjs` 根據既有完整 `mechanism-results-v2.json` 產生；原 `mechanism-results-v2.js` 在本機無法讀取，因此保留原檔而另建等值載入檔。重建工具驗證完成狀態、35 個成果校驗碼、22 項下載及新舊資料物件一致，不改資料、原檔權限或研究結果。

## 教學算例

四個節點 i、B、C、D，初始值 [2, 1, 4, 3]。固定圖在兩個時間點都使用 B、C；動態圖在 t₁ 改用 B、D。可選五種模型、修改任一節點特徵，並以「輸入 → 權重 → 加總 → ReLU」四步檢查更新。

- W = 1、b = 0；MLP 不使用鄰居。
- GCN 採含自迴路的對稱正規化：目標 degree = 3、相連鄰居 degree = 2，因此係數為 1/3 與 1/√6，並不是算術平均。
- GAT 使用 a = (0, 1)、LeakyReLU 負斜率 0.2 與 softmax。
- 數值是未訓練的節點表示，不是事件機率或模型表現。

## 第二版網頁與資料接入

網頁 Simulation 已改為第二版機制研究，依序為研究問題、生成器、圖與模型資訊、A/B/C/D 情境、可觀測診斷、公平比較、結果及真實資料轉譯。中英文均對齊實際程式；主要 synthetic task 是一步 regression，Stage A smooth_signal／Stage B tau 是同時點 denoising，classification 為另列的 robustness。實際 1/3/6 小時高濁度事件仍需外部驗證。

結果分七區：Stage A 機制檢查、五條反應曲線、三張相圖、200 條件多軸研究、樣本數、穩健性與 200/400-epoch 敏感度。正式批次已於 2026-10-05 23:41（台北時間）完成 284 工作、7,240 次模型評估；Stage A 的 150 次評估另列。2026-10-06 核對後，完整數據、14 張科學 PNG、22 個下載連結及七區中英文解讀已接入 `dist/data/mechanism-results-v2.js`，不混用第一版分數。

匯出器把分數、CSV、科學 PNG 與來源校驗碼一次接到網站，不重新訓練；以下命令可重新匯出這份已完成批次：

```powershell
& algorithm/.venv/Scripts/python.exe scripts/export_mechanisms.py --batch algorithm/results/v2_full_suite_recovered_20261003 --stage-a algorithm/results/v2_stage_A --destination dist/data
```

匯出器核對完成標記、預定工作、完整配對 seeds、模型與來源，拒絕 partial／smoke。腳本只讀正式模擬結果，不修改權重或背景源碼；匯出的 JavaScript 讓 GitHub Pages 與離線 `file://` 開啟都可使用。

`dist/data/mechanism-discussion-v2.js` 現已填入七個區塊的 `zh`／`en` 解讀，可逐段修訂。主要結論與限制也整理在 `algorithm/results/v2_full_suite_recovered_20261003/RESEARCH_FINDINGS_20261006.md`。δ 的收益非單調、帶雜訊目標的 H 非單調、Static 與 Dynamic 的 R̂ 關聯方向不同，以及 R̂ 的估計誤差都保留於討論。200／400 epochs 結果分開展示，不事後依 test 分數挑預算。診斷相圖尚未成為經外部驗證的選模規則。

完成後檢查發現三個 condition hashes（`036689513890`、`2e8395032995`、`50028716e038`）被 pandas 的 CSV 型別推斷改成數值。`suite.read_metrics_csv` 現明確保留字串，並以 `scripts/repair_v2_reports.py` 從完整 per-job CSV 重建報表。7,240 份原始 `result.json` 的測試分數逐一核對一致；權重、模型預測、訓練 checkpoint 與 frozen source 都保留。原始報表在 `reports_before_id_repair_20261006`，修正來源與校驗碼在 `report_repair.json`，公開 provenance 也記錄報表修正。修正只重建報表，沒有重新訓練；該 repair 腳本拒絕覆寫既有原始報表備份。若要以修正後程式重新訓練，使用新的 output 目錄，不重用這份完成批次。

第一版 `overnight_20260930` 的 945 工作、腳本與下載原樣保留，主頁不再載入其結果腳本。它與第二版使用不同生成器、任務及模型輸入，兩版不得混合。原第一版匯出命令為 `scripts/export_simulation.py`，只供該封存版本使用。

真實案例仍只做預測。事件需固定濁度門檻、機率門檻、事件合併、冷卻、缺測及配對方法；取水、供水與淨水作業的決策最佳化不在專題範圍。

## 驗證與來源

2026-10-04 第二版網頁改寫已檢查：五章 zh/en 渲染、十個 Simulation 公式的 KaTeX 語法、七個結果區、反應曲線／相圖選單、390px 響應式及離線開啟。另以測試頁接入實際資料 schema，驗證分數表、配對表和科學圖表能顯示；測試頁只在 tmp，不發布 partial 結果。

第二版匯出器的 12 項完整性測試通過，另包含已驗證／未驗證的報表修正 provenance；編號回歸測試也通過，涵蓋前導零、科學記號與超大 exponent 的 hash。正式批次已通過 `--check-only` 的所有完成數、配對種子、來源、CSV／JSON 分數及圖表覆蓋檢查。正式網頁只接入核對後的完整結果，未使用 tmp 測試頁代替研究資料。

```powershell
& algorithm/.venv/Scripts/python.exe -m unittest discover -s scripts -p test_export_mechanisms.py -v
& algorithm/.venv/Scripts/python.exe scripts/export_mechanisms.py --check-only
& algorithm/.venv/Scripts/python.exe -m unittest discover -s algorithm/tests -p test_report_identifiers.py -v
node tmp/check_website_formal_v2.cjs
```

`--check-only` 只讀取結果；正式批次未完成時會明確拒絕。正式頁檢查已通過，涵蓋公開下載 SHA256、五條曲線、三張相圖、全部中英文解讀、Stage D 200 組條件、診斷關聯數值與 390px 顯示。原有的 2026-09-30 驗證紀錄如下。

已核對五種教學算例的純數學計算。2026-09-30 的結果接入另檢查：945 工作數值、敘述中的條件平均、下載檔 checksum、中英文切換、情境／指標切換、配對區間與手機版顯示。結果來自已完成的封存批次，不從示意滑桿生成。

模型定義連結至原始 GCN／GAT 論文與作者說明；濁度背景連結至 USGS 與水利署官方頁面。資料入口不代表已取得可用的歷史資料。

KaTeX 的原始授權保留於 `dist/assets/vendor/katex/LICENSE`。Rozias logo 延用你的既有 Spectral Graph Lab 資產。
