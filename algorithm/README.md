# 演算法與研究執行

> **目前範圍（2026-11-10）：** 本專題以第二版模擬研究為完整主體，不再執行水庫實際案例或實測外部驗證。下方第一版及實測資料工具保留為歷史紀錄，並非現行待辦。現行範圍見 [RESEARCH_SCOPE.md](../RESEARCH_SCOPE.md)，模擬方法見 [MECHANISM_V2.md](MECHANISM_V2.md)。

新版機制研究已建立獨立程式：方法見 [MECHANISM_V2.md](MECHANISM_V2.md)，單一階段執行 `run_mechanisms.py`。完整 B/C/D、sample-size、robustness 與收斂檢查已具備背景佇列：[SIMULATION_SUITE.md](SIMULATION_SUITE.md)；執行 `./algorithm/start_simulation_suite_background.ps1`。第一版資料下載、事件預測與封存批次仍由下列原程式管理；兩版結果不混用。

核心流程已實作：資料下載與稽核 → 因果時間對齊 → 五模型訓練 → 時間與事件評估 → 模型選擇規則 → 外部套用 → CSV、圖表、HTML 與網站 JSON 匯出。

研究範圍、數學定義、固定設計、資料缺口及限制請先讀 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。不要將 smoke 或 pilot 的輸出當成完整研究結論。

## 使用環境

目前專案已有 Python 3.12 虛擬環境 `algorithm/.venv`。以下指令均在專案根目錄執行：

```powershell
& algorithm\.venv\Scripts\python.exe -m unittest discover -s algorithm/tests -v
& algorithm\.venv\Scripts\python.exe algorithm/run_research.py simulate --config algorithm/configs/smoke.json --output algorithm/results/my_smoke
```

其他電腦先以 Python 3.12 建立虛擬環境，再安裝 `algorithm/requirements-lock.txt`。正式結果需保留 Python、套件版本及 CPU/GPU 設定。PyTorch 預設 CPU、2 threads。

## 第一版背景執行封存

```powershell
& algorithm\.venv\Scripts\python.exe algorithm/run_batch.py --output algorithm/results/overnight_20260930
```

這個指令預設啟動 `overnight.json`：21 情境 × 3 seeds × 3 horizons × 5 模型 = 945 模型工作；每個工作比較 2 個候選。完成後才檢查真實資料並嘗試套用模擬規則。程式會先保存源碼及設定快照、跑快照測試，再進行訓練。

- 進度：`algorithm/results/overnight_20260930/status.json`
- 訓練紀錄：同資料夾的 `simulation.log`
- 完成後入口：`index.html`
- 模擬：`simulation/report.html`、`metrics.csv`、`summary.csv`、`paired_differences.csv`、`selection_rule.json`、`website_results.json`
- 真實資料：`real/report.html`，可能是結果，也可能是明確的資料不足報告。
- 重跑同一指令會使用原有快照與完整模型 checkpoint 續跑。請先確認原批次已結束；鎖檔防止同時寫入。
- 電腦關機或睡眠期間不會繼續計算。沒有修改你的電源設定。

目前 NX08 固定時間切分下的驗證與測試集沒有 >1000 NTU 事件，真實模型比較將標示 `insufficient_data`。背景批次可正常完成為 `complete_with_data_gap`，這不表示外部驗證完成。

## 正式完整網格

```powershell
& algorithm\.venv\Scripts\python.exe algorithm/run_research.py simulate --config algorithm/configs/research.json --dry-run
& algorithm\.venv\Scripts\python.exe algorithm/run_research.py simulate --config algorithm/configs/research.json --output algorithm/results/full_v1
```

完整網格包含 100 個有效情境、5 seeds，共 7500 模型工作與 30000 個候選 fits，可能需要多天。第一輪因子篩選完成後，應檢查收斂、區間及參數範圍，再决定論文中需要哪些補充實驗；不是只看某一個 seed 的最高分。

## 資料下載、品質與事件可用性

```powershell
& algorithm\.venv\Scripts\python.exe algorithm/fetch_data.py --sources turbidity reservoir rain_stations
& algorithm\.venv\Scripts\python.exe algorithm/run_research.py audit
& algorithm\.venv\Scripts\python.exe algorithm/inspect_readiness.py
```

已保存 427,373 筆水利署原始濁度資料及 checksum。API 的近期水情與 2021 年濁度不重疊，不能直接拼接。資料說明和每月正例數分別在 `data/audit/data_audit.json` 與 `label_readiness.json`。

取得額外歷史資料後可用 `data/observations_template.csv` 的長表欄位。實測設定的 data 區塊加入 `observations_csv` 路徑與 `graph_variable_by_station`，詳見研究規格。程式檢查單位、時間、發布延遲與品質，標籤不補值。

```powershell
& algorithm\.venv\Scripts\python.exe algorithm/run_research.py real --config algorithm/configs/real_research.json --selection algorithm/results/full_v1/selection_rule.json --output algorithm/results/real_v1
```

## 程式模組

| 模組 | 責任 |
|---|---|
| config / graphs / simulation | 生成器、獨立亂數流、burn-in、邊漂移 |
| sources / real_data / csv_data | 公開來源、原始封存、時間對齊、品質、CSV |
| estimators / indicators / descriptors | 圖估計與只使用訓練期的可觀測指標 |
| protocol / models / training | 相同時間切分、五模型、調參、early stopping |
| evaluation | 視窗分類、回歸、事件配對、提前量、block bootstrap |
| selection | 按生成情境分組的規則評估與外部拒絕選擇 |
| experiment / reporting | 續跑、配對比較、圖表、結果 JSON 與 HTML |
| run_batch.py | 冻結快照、背景順序執行、鎖檔、狀態與日誌 |

原始 `run_demo.py` 保留作生成器示例。主研究請用 `run_research.py` 或 `run_batch.py`。

## 網站後續接入

網站 `dist/` 保留原有設計，本輪先完成研究程式。研究 JSON 和 PNG 已有輸出介面，後續將接入 Simulation 與 Event Case Studies；資料不足的部分保留有來源的說明，不能填入虛構預測。

`results/`、原始資料及虛擬環境已排除 Git。發布網站時只挑選經確認的聚合結果、圖表、規格與來源資訊；不要把所有原始 records、模型權重與私有資料放到 GitHub Pages。

