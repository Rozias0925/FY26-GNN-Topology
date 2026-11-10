# 真實濁度回歸：保守背景執行

本次先執行原 2021 年設定的輔助 regression task，預測 NX08 未來 1／3／6 小時濁度。分類仍是專題的主要研究目標，但 validation 沒有高濁度正例；本次不改門檻、不改時間切割、不以回歸結果宣稱完成事件預警驗證。

## 固定比較與檢查

- MLP、Static-GCN、Static-GAT、Dynamic-GCN、Dynamic-GAT。
- 每站 24h 歷史：2 個 log-NTU 特徵＋2 個 missing masks，flatten 成每站 96 維。保留 node 維度，僅以 NX08 的 scalar target 計算 loss。
- 相同 60%／20%／20% chronological split，purge 跨界 future targets；target 不插補。
- 使用已驗證的 hourly panel 與 causal graph cache，重新核對輸出 SHA-256；不重新載入 220 萬筆原始資料佔用大量 RAM。
- 回歸單獨處理 point-target validity，不因中間某小時缺少 event label 就刪掉已觀測的 t+h point target。
- 同一資料、5 個 initialization seeds、2 widths × 2 learning rates、120 epochs 上限、patience 15，共 **75 個 model jobs、300 次候選 fitting**，順序執行。
- 以 validation 的 training-standardized log1p(NTU) MSE 選模。全部候選選模結束後，才開始 test model scoring。
- 先執行 `tests/test_real_benchmark.py` 的小型 synthetic checks，包含五模型 shape、NX08 loss、test poisoning、時間解析度、preprocessing leakage、完整微型 pipeline、resume 與資源 gate。任何檢查失敗即停止，不啟動真實資料 fitting。
- 修正 pandas 3 可能保留 us/s resolution 的相容性問題：時間間隔檢查先明確轉為 ns。

## 保護其他工作

```powershell
& ./algorithm/launch_real_benchmark.ps1 -ConfigPath ./algorithm/configs/real_benchmark_regression.json -OutputPath ./algorithm/results/real_regression_20261020 -WaitForResources
```

啟動器先量測資源；不足時只啟動使用 Python 標準庫的輕量 helper。helper 使用 **Idle priority、單一 logical CPU**，不載入 numpy、torch 或模型。每 30 秒檢查一次，須連續三次（各量測約 2 秒）符合 **可用 RAM ≥1.5 GiB、整體 CPU ≤70%** 才進入測試與 preflight。等待超過 24 小時會記錄 stopped，不會自行放寬門檻。

2026-10-30 已以 Windows Job Object 將專用測量程序的 private commit 限制為 **768 MiB**，實際執行 12 項小型檢查（含微型端到端 fitting），12 項全數通過、測試耗時約 32 秒。峰值 working set 約 **370 MiB**、private commit 約 **361 MiB**；測量期間系統可用 RAM 最低約 1.109 GiB。測量原始紀錄見 `tmp/real_memory_profile_20261030_attempt2/profile.json`。原先 4 GiB 是自訂保守門檻，不是模型的硬性需求，因此依此次量測改為 1.5 GiB 啟動門檻，並加上程序記憶體上限。

上述量測涵蓋小型 synthetic 檢查，**不是完整真實資料訓練的峰值量測**。依已驗證的 N=3、T=7320、每站 96 維 float32 歷史輸入及 packed graphs 計算，三個 horizon 的主要輸入約不超過 **26 MiB**，作為評估記憶體餘裕的依據；正式 fitting 的實際峰值仍須由執行紀錄確認。

數值階段仍是 **單 process／單 worker／單 logical CPU／所有數值函式庫 threads=1／CPU-only**。主 runner 在載入數值套件前，對自己的程序套用 Windows Job Object **768 MiB private-commit 硬限制**。若整體 CPU >85% 或可用 RAM <1.0 GiB，在下一個 batch 邊界暫停；RAM ≥1.25 GiB、CPU ≤70% 後才繼續。若可用 RAM <0.75 GiB，則退出自己的數值程序以釋放記憶體，保留已完成候選的 checkpoints；尚未完成的候選留待續跑。單純暫停並不會釋放已載入的套件，因此另設此退出條件。所有 priority、affinity、記憶體限制及 environment 設定只作用在本次研究程序；不調整、終止其他程式，不使用 GPU。

直接執行 `run_real_benchmark.py` 也會先做自己程序的優先權與 RAM/CPU gate，再載入數值套件。模型、輸入及 Python 本身仍需要資源；此政策是減少干擾，不是承諾完全零影響。

## 狀態與停止

- `launcher_status.json`：waiting_for_resources／running／completed／failed／stopped、自己的 PID、實際 CPU/RAM 量測及日誌位置。
- `.launcher/*.stdout.log`、`.stderr.log`：等待、測試與訓練訊息。
- `status.json`：重型階段開始後的 tests／preflight／training／evaluating／complete 與 model/seed/horizon/candidate 進度。
- `tests.log`、`tests_passed.json`：實際數值測試結果；尚在等候時不會存在。
- `preflight.json`、`manifest.json`：正式 regression run 的設定、資料、程式、環境及計畫 identity；尚在等待時不會存在。

若要只停止本次工作，在其輸出資料夾建立 `STOP_REQUESTED` 空檔即可；等待 helper 或 batch 邊界會自行退出，保留已完成候選。不要終止所有 python 程式。

重新執行時必須沿用相同資料、設定、程式與環境；已完成候選會核對 weights SHA 後重用。程式變更時，使用輸出目錄 `source_snapshot/run_real_benchmark.py` 搭配原本未變更的 `algorithm/configs/real_benchmark_regression.json` 和相同 `--output`；不把不同版本的候選混成同一份結果。重新開始前先移除自己輸出目錄的 `STOP_REQUESTED`。

## 完成後

輸出 `metrics.csv`、`model_summary.csv`、`paired_contrasts.json`、每個 model/seed/horizon 的預測與 weights，以及 `REPORT.md`。`complete.json` 表示此次 regression 比較完成，classification 與正式外部有效性仍另行標記未完成。

報告以原始 NTU 的 MAE／RMSE、原始與 log-NTU R² 呈現；配對比較用共同 test origins、24h moving time blocks，不跨缺時。五個 seeds 是同一資料集的初始化變異，不是五份獨立水文資料。Persistence 僅在 current observation 有效的共同 origins 比較，另列排除數，避免不同樣本分數直接相減。

三個測站仍在 synthetic N=100–1000 支援範圍外；GP17 是下游，相關圖不是已驗證的河道物理圖；同期歷史水文 predictors 與發布延遲仍未補齊。本次結果只作探索性回歸比較。
