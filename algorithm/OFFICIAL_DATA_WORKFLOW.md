# 官方資料下載與分析

在專案根目錄（`Static and Dynamic Graph Construction for GNNs`）執行。以下路徑是命令的參數，相對於該目錄。Python 使用既有 `algorithm/.venv`，不需 API key。

```powershell
$env:PYTHONUTF8='1'
& algorithm/.venv/Scripts/python.exe algorithm/fetch_official.py --output algorithm/data/official/20261011 --sources turbidity reservoir rain_stations river_stations river_level catchment_rain --max-pages 8000
& algorithm/.venv/Scripts/python.exe algorithm/analyze_official.py --raw algorithm/data/official/20261011 --config algorithm/configs/official_analysis.json --output algorithm/data/analysis/20261011
& algorithm/.venv/Scripts/python.exe algorithm/profile_official_history.py --raw algorithm/data/official/20261011 --audit algorithm/data/analysis/20261011 --output algorithm/data/analysis/20261011/full_history
& algorithm/.venv/Scripts/python.exe algorithm/verify_official.py --raw algorithm/data/official/20261011 --analysis algorithm/data/analysis/20261011 --output algorithm/data/verification/20261011.json
```

重新抓取另一日的快照時，使用新的日期資料夾。同一資料夾會讀取已完成的壓縮頁與雜湊後續傳，不會把快取說成新下載。若最後 `download_complete=false`，分析會拒絕把未完整濁度當作完整研究資料。

## 下載的官方來源

| 資料 | 官方目錄 / API | 用途 |
|---|---|---|
| 石門水庫濁度 | https://data.gov.tw/dataset/58688 | 主要觀測與 target |
| 水庫水情 | https://data.gov.tw/dataset/45501 | 雨量、進流量、水位；需核對期間 |
| 雨量站基本資料 | https://data.gov.tw/dataset/32729 | 測站 metadata，不是雨量序列 |
| 即時河川水位 | https://data.gov.tw/dataset/25768 | 官方即時快照，不保證歷史 |
| 河川水位測站站況 / 水庫集水區雨量 | https://opendata.wra.gov.tw/api/v2/openapi.get | 已核對 endpoint ID 與名稱 |

原始內容在各 source 子目錄的 `page_*.json.gz`，`manifest.json` 留存 URL、擷取時間、筆數、SHA-256 和授權。`metadata/` 保存當次官方目錄與 OpenAPI；程式先核對 endpoint 身分。一般資料 API 是分頁 list；集水區雨量是 FEWS-PI `timeSeries`/`header`/`events` 的單次快照，保留 Big5 編碼原始位元組、timeZone、type、timeStep、missing-value sentinel 與 quality flag。程式不把 HTML 維護頁或錯誤 dict 当成空資料。

水利署來源採政府資料開放授權條款第 1 版（https://data.gov.tw/license）。保留官方來源標註；API 分頁結束僅證明當次傳輸結束，未證明所有年份皆完整公開。

## 分析輸出

`ANALYSIS_REPORT.md` 為可直接閱讀的分析報告。其他機器可讀檔案：

- `station_inventory.csv`、`station_metadata.csv`、`data_audit.json`：原始範圍、缺失、零值、負值、重複、間隔，以及已驗證測站位置。
- `source_inventory.csv`、`official_*.csv`：官方輔助資料與期間重疊。所有國家級站況仍保留；reservoir 與 catchment rain 的整理 CSV 限石門。
- `hourly_turbidity.csv`、`hourly_panel.npz`、`panel_metadata.json`：未插補的 hourly last/peak target；輸入為 log1p(last/peak)。
- `forecast_labels.csv`、`label_readiness.csv`：1／3／6 小時 point target 和未來區間超標事件；缺失、purged windows 與 class counts 明列。
- `events.csv`：超標片段、峰值、觀測 duration、onset/end 是否有完整邊界。缺失會中斷事件，censored 事件不當成完整事件。
- `causal_graphs.npz`：校準 prefix static graph、過去 rolling-window dynamic graphs、timestamps 和 node order。
- `training_diagnostics.json`：training-only S_X、S_G、H、R_hat、D_struct、D_pred，shuffled negative control 與內部時間 fold 邊界。
- `baseline_metrics.csv`、`baseline_predictions.csv`、`baseline_paired_comparisons.json`：persistence 與 training median/prevalence 對照，MAE/RMSE、window metrics 與 paired time-block CI。
- 三張 PNG：hourly 序列、月事件與 missing coverage、校準圖。
- `analysis_manifest.json`：完成旗標、來源、設定、程式與輸出的 SHA-256。下游引用前確認 `completed=true` 並驗證雜湊。

`full_history/FULL_HISTORY_READINESS.md` 另外盤點三站共同的全部歷史期間；區間由 timestamp coverage 選定，不依 target 值、事件數或模型分數選定。附各年的覆蓋／超標小時、事件片段、完整邊界片段及 1／3／6h sample checks。這是候選資料盤點，不替換原 2021 年設定、不擬合模型、不自動核准正式外部驗證。

`verify_official.py` 重新驗證原始頁內容 SHA-256、逐頁與總筆數、官方 metadata、兩份分析的輸出 SHA-256 以及來源 identity；它只讀取上述成果，不替內容改寫校驗碼。

## 固定規則與限制

設定集中在 `configs/official_analysis.json`。本次分析沿用 2021-01-01 到 2021-11-02（不含右端）、NX08、1000 NTU、60%/20%/20% chronological split；新增官方歷史資料仍完整保存與盘點，不依 test 的事件數挑選期間或 threshold。若另訂更長研究期間，先明確記錄設定，再用新的分析輸出目錄，不能冒充原設定的結果。

每小時 `(t−1h,t]` 至少六筆觀測；point regression 為 t+h 該小時最後觀測。事件為未來 `(t,t+h]` 任何 hourly peak **嚴格大於** threshold；全窗口觀測完整才標 0/1。不插補任何 target；跨 split 的 future targets purge。Regression 與 classification 分開檢查可用性，單類別 test 的 AUROC 或零正類的 AP/recall 不造值。

Static 圖只用前 168h calibration，首個 forecast origin 在 calibration 結束後；dynamic 只用截至當下的 rolling 168h，每 6h 更新。圖是 **無向 lagged-correlation 估計圖**，不是官方物理圖；未使用生成圖、target 未來值或 validation/test 建 static graph。訓練 probe 的 inner fold 另外重新估計 preprocessing，並 purge forecast horizon，避免 inner validation leakage。

NW12＝龍珠灣；NX08＝壩前浮台；GP17＝後池堰排砂道。GP17 是下游，不能改稱上游。NW22 的位置與深度尚未在當次目錄核實，不投入指定三站 panel。

資料中的水庫累積雨量不能直接當 hourly rain；FEWS accumulation semantics/flag 意義須核對。即時 2026 水情與 2021 濁度不能合併。發布時間延遲未核實，offline 可用觀測不等同 operational early-warning 提前量。三站資料位於 synthetic N=100–1000 之外；此 diagnostics adapter 的 log-NTU、24h history 與 synthetic p=8 task 不同，不能直接按數值宣稱 phase-map winner。

這份程式是完整的資料分析與 baseline 流程，**不是五模型正式外部驗證完成**。專題仍需可驗證的同期 predictors、足夠事件、記錄 simulation-based 預期，才比較五模型 held-out 行為。網頁可用本流程輸出的描述圖表與限制，不應把 readiness、baseline 或缺少事件的 test 寫成預警成功。

## 檢查

```powershell
& algorithm/.venv/Scripts/python.exe -m unittest discover -s algorithm/tests -p test_official_analysis.py -v
& algorithm/.venv/Scripts/python.exe -m unittest discover -s algorithm/tests -p test_pipeline.py -v
```

核心測試涵蓋：Big5 FEWS-PI、下載失敗與重複頁、HTTP 200 HTML 重試、缺失 future label、不含現在小時、strict threshold、時間 purge、missing-boundary event censoring、changing test values 不改 training diagnostics、paired bootstrap。
