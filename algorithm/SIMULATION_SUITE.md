# 第二版完整模擬執行

目前有效批次是 `results/v2_full_suite_recovered_20261003`，即時位置亦保存於 `results/v2_latest.json`。原 `v2_full_suite_20261003` 在 I/O／failure logging 錯誤後中斷並保留；已完成權重經檢查後移入新批次接續，不重新訓練。進度與修復紀錄見 [2026-10-03 進度](results/v2_full_suite_recovered_20261003/PROGRESS_20261003.md)。

恢復批次的同設定續跑指令：

```powershell
& ./algorithm/start_simulation_suite_background.ps1 -Output algorithm/results/v2_full_suite_recovered_20261003
```

程式會從既有 manifest 讀取 recovery source；active lock 會阻止重複執行。跨版本 recovery 只接受相同設定、完全相同的 computation modules 與 fit/score/design AST，保留原 source checksums 及報表 evaluation_source_sha256。

本批次涵蓋研究計畫的剩餘模擬程式與無人值守佇列。它不等於已完成所有模擬結果，也不等於石門水庫資料或外部驗證已完成。第二版主方法見 [MECHANISM_V2.md](MECHANISM_V2.md)。

## 範圍與固定設定

| 區塊 | 內容 | 主要模型與預算 |
|---|---|---|
| Stage B | λ、κ、τ、δ、η 的五條 one-axis response curves | 五模型；200 epochs 上限 |
| Stage C | graph signal/relevance、drift/noise、smoothness/signal 三組 interaction grids | 五模型；200 epochs 上限 |
| Stage D | 200 個 Latin-hypercube conditions | 五模型；200 epochs 上限 |
| Sample size | N=100/300/500/1000；κ=0 | 五模型；200 epochs 上限 |
| Robustness | Regular controls、heterogeneous degree、SBM、kNN、σ=.1/.5/1、binary classification、query-dependent relevance | 原五模型及兩個 GATv2；200 epochs 上限 |
| Convergence | 重新執行六個 Stage A conditions | 五模型；只把 max epochs 改為 400；patience、candidate grid、split 不变 |

所有正式區塊使用 5 個資料種子 7/19/31/43/59，同一資料內的比較是 paired；共用 chronological split 與 training-only preprocessing/diagnostics。主設定 N=500、T=40、p=8、d=10；sample-size 與 graph-family experiments 的指定介入為例外。四組 tuning candidates = width 32/64 × learning rate .001/.003。

Primary 200-epoch results 與 400-epoch sensitivity 分開保存。不能把延長訓練後的最佳分數偷偷取代原先結果；sensitivity 只分析固定預算改變的影響。每個 candidate 記錄 epochs_run、best_epoch、stopped_reason、budget_exhausted；達上限的結果會在報表中標記，不能假設 early stopping 已達穩定點。

## 工作量與執行順序

去掉同一 primary config 裡的重複 condition後，主研究共 264 conditions、6,600 model evaluations；convergence 共 150 evaluations；robustness 共 490 evaluations。總計 **284 個佇列工作、7,240 次模型評估、28,960 次參數候選評估**。Stage D 仍完整使用 200 conditions，沒有為了 overnight 改小正式 N、T、種子數或 condition count。

同一 condition/seed 若在 B/C/sample-size 重複引用，只訓練一次、保留每個實驗 reference。若動態圖在所有 snapshots 與 static consensus 完全相同，Dynamic 與 Static 的同架構、同初始化、同候選資料張量完全相同，可引用已完成的 Static candidates。result.json 明確記錄 training_reuse，candidate_fits_executed=0；獨立模型訓練等價性已用測試驗證。故參數候選評估數與實際 optimizer fits 不相同，報表另計實際 fits，不把省略的重複計算冒稱為新訓練。

順序：五條 B 軸輪流排點 → Stage A 400-epoch sensitivity → sample size → robustness → 剩餘 C → D。在其他階段需要相同資料與設定時，可提早產出那些 reference；不以某個假說是否成立作執行門檻。

兩個獨立 Python workers，每個模型 2 CPU threads；不使用共享 random-state threads 訓練。這台電腦有 8 logical processors，此設定預留部分系統資源。完整實驗可能需要多天；優先順序讓早期即可看到五條 B 曲線，不承諾全部在一個睡眠時段完成。

## 啟動與續跑

在專案根目錄：

```powershell
& algorithm/.venv/Scripts/python.exe algorithm/run_simulation_suite.py --dry-run
& ./algorithm/start_simulation_suite_background.ps1
```

預設輸出 `algorithm/results/v2_full_suite_20261003`。啟動器使用隱藏的背景程序，工作不依賴聊天回合保持開啟。主程序在執行期間提出暫時的 Windows keep-awake request，結束時撤回，不修改永久電源設定；關機、斷電或手動睡眠仍會中止／暫停運算。

同設定、同源碼再次執行可續跑。源碼或設定不同會拒絕混批，必須用新輸出目錄。suite.lock、run.lock 防止重複工作；程式只有在確認持有 PID 不存在或已終止時才移除殘留的單一 lock file，保留 checkpoints。已完成條件會跳過，未完成模型會續跑；中斷當下尚未完成的 model candidates 會重新訓練。

每個 job 的例外寫入 queue_error.json 和 worker.stderr.log，其他獨立 jobs 仍可進行。若任何 job 失敗，全套不寫 complete.json，status 為 failed；不能把錯誤當作完成或忽略缺失資料。

## 早上查看哪些檔案

- 根目錄 `status.json`：完成模型數、正在執行的條件、分組進度與 failures。
- `queue_plan.json`：全部條件及 B/C/D 的 references；`events.jsonl`：每個工作完成或失敗紀錄。
- `reports/main/`：metrics.csv、paired_contrasts.csv、五條 response curves、三張 phase diagrams、sample-size figures、convergence_flags.csv 與 website_results_v2.json。
- `reports/robustness/`：七模型、classification AP/AUROC、穩健性 contrasts 與 figures。
- `reports/convergence/`：400-epoch 分數；budget_sensitivity.csv 與 budget_sensitivity_paired.csv 比較原 Stage A 200-epoch 分數。
- `jobs/<group>/<condition>/datasets/<condition>/seed_<seed>/<model>/`：weights.pt、predictions.npz、training.json、result.json。

每個條件的全部種子與模型完成後，即刷新彙整與圖表，不必等待最後一個 D condition。報表 status 與 website JSON 帶 partial/complete 狀態與已完成／預定 conditions 數；只有相应 complete.json 出現才算整組完成。Partial 圖表可以檢視，不能當成完整 phase map；smoke 圖表一律標明 execution check only。

## 驗證紀錄（2026-10-03）

- 既有 V1/V2 57 項測試與 8 項佇列／budget／等價性／跨階段匯出測試通過，共 65 項。
- 完整佇列小規模執行：30 jobs、178 evaluations，涵蓋 B、D、sample-size、convergence、14 種 robustness，全部完成、無 queue errors、worker stderr 為空。
- C 與 B 同一 condition 的跨階段 references、phase diagrams、避免重複訓練已另做實際執行測試。
- 完成後再次執行 smoke queue：178 個權重 checkpoint 的修改時間全部不變，證明續跑沒有重新訓練完成模型。
- Source SHA256（此版本）：`6bf74ff61cdfefce21fa7c69f9566fe9ff4397d4ae0865545ac7fbeecaa7eafc`。原 Stage A 和第一版封存資料保留各自 source snapshot，網站沒有改寫。
