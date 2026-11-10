# 第二版程式驗證紀錄

研究入口：`run_mechanisms.py`；方法、假設與正式執行指令：[MECHANISM_V2.md](MECHANISM_V2.md)。

本輪完成獨立 V2 機制生成器、snapshot 模型、training-only diagnostics、Stage A/B/C/D/sample-size/robustness 設計、paired contrasts、seed bootstrap、圖表與網站 JSON、外部 regression expectation freeze/compare 介面。

核心源碼 SHA256：

`d4f431a19f2720e03bfb979395787e9bef40c65b26d5e3b3c800c99a98f48167`

## 驗證結果

- V2 的 19 項測試通過；既有 V1 的 38 項測試通過。檢查包含 target-time split、future exclusion、train-only calibration、regular normalization、attention entropy、degree/density preservation、noise/target separation、permutation equivariance、fixed-topology equality、test-label tuning isolation、paired interaction、CSV round-trip numeric types、preregistration cutoff/immutability/all-five requirement。
- `results/mechanism_v2_smoke_20261002`：Stage A 小規模 60 model evaluations 完成。
- `results/mechanism_v2_pipeline_verified_20261002`：Stage A/B/C 小規模 640 個 unique model evaluations 完成；900 個實驗 reference rows、810 個 main contrast rows；五條 response curves 與三張 observable phase maps 匯出成功。
- `results/v2_extended_verification_20261002/run`：Stage D 的三個 Latin-hypercube conditions 加 robustness，共 105 個小規模 evaluations，包含 classification 與 optional GATv2；另外輸出 GATv2-minus-GAT contrasts。
- Report-only CSV reload / re-export 已實際執行，修正 mixed regression/classification scores 被讀為字串的問題；JSON 中 scores 保持 numeric。
- 圖表已視覺檢查，所有 smoke 圖表明確標示 execution check only，不作假說證據。
- 各批次 manifest/source 保留開始執行時的快照；後續報表格式與外部驗證 guard 的修正不更改已保存的模型分數。源碼變更後正式執行應使用新的輸出目錄。

## 正式大小的生成器與 probe 抽查

輸出：`results/v2_generator_review_20261002.json`。使用 N=500、T=40、d=10、seed=7、正式 80-epoch gating probe；**沒有訓練正式五模型**。

| 機制 | S_G | R_true | R_hat | D_pred |
|---|---:|---:|---:|---:|
| Feature-only | -0.0008 | 0 | 0.0044 | 0 |
| Uniform neighbors | 0.7873 | 0 | 0.0001 | 0 |
| Selective neighbors | 0.3842 | 0.6965 | 0.6156 | 0 |
| Smooth signal | 0.0326 | 不適用 | 0.0033 | 0 |
| Predictive dynamics | 0.7541 | 0 | 0.0000 | 0.8273 |
| Noisy dynamics | 0.2727 | 0 | 0.0006 | -0.5104 |

此抽查支持 generator/probe 可以区分指定機制；它是一個種子的稽核，不能證明各模型相對優勢的曲線成立。Selective 情境的 uniform-summary S_G 與 R_hat 誤差應在正式機制分析中保留。

## 正式批次進度與待完成工作

2026-10-02 再次執行全部 57 項測試，全部通過；formal dry-run 與背景啟動器語法檢查亦通過。正式 Stage A 已於台灣時間 20:57:32 啟動，輸出目錄為 `results/v2_stage_A`，預定 6 個情境 × 5 個 seeds × 5 個模型，共 150 個 evaluations、600 個 tuning candidate fits。使用正式 N=500、T=40、epochs=200、patience=25 設定，核心 SHA256 與上列紀錄相同。

背景執行指令：`./algorithm/start_mechanisms_background.ps1`（在專案根目錄）。`launcher.json` 保存啟動程序與日誌路徑，執行期間 `run.lock` 保存實際 worker PID；`preflight.json` 保存本次執行前的檢查摘要；`status.json` 與個別模型 `result.json` 記錄實際完成進度。只有 `complete.json` 出現且 status 為 complete 才代表整批完成。

2026-10-03 確認本批次已於台灣時間 2026-10-02 22:32:15 完成。核對 150 個唯一模型結果、30 份 paired datasets、600 個 tuning candidate records 與有限的 test scores；stderr 為空。正式六情境結果與推論限制見 [Stage A 結果摘要](results/v2_stage_A/RESULTS_SUMMARY_zh-TW.md)。此批次支持指定極端機制的相對行為，但不能取代 Stage B response curves；部分模型最佳 epoch 達 200 上限，後续仍須確認收斂敏感度。未改動核心源碼、保存的模型分數或網站。

正式 B/C/D 的 response curves、multi-axis map、sample-size 與 robustness 推論仍需依序執行並分析。石門水庫的資料缺口、1/3/6 小時事件任務與 held-out 外部驗證尚未完成；目前 external adapter 是 regression 的初始轉譯介面。

第一版封存結果與網站內容保留。第二版 smoke／generator audit 不會自動寫成網站的正式研究結論。

## 2026-10-03 完整模擬佇列

剩餘 B/C/D、sample-size、14 個 robustness conditions（含 GATv2）與獨立 400-epoch Stage A sensitivity 已補齊並加入可續跑的雙程序背景佇列。跨階段相同 primary condition 只訓練一次；完全相同固定圖的 Dynamic 模型可引用 Static candidates，並明確記錄 training_reuse 和實際 optimizer fits。每個 candidate 新增收斂預算標記；增加 sample-size/robustness 圖表與逐條件匯出報表。

此版本核心 SHA256：`6bf74ff61cdfefce21fa7c69f9566fe9ff4397d4ae0865545ac7fbeecaa7eafc`。65 個唯一測試通過；30-job／178-evaluation smoke queue 全部完成、無 queue errors；再次執行後 178 個權重 checkpoints 修改時間不變。B/C 共用 condition 與三種 phase plots 另有實際測試。先前正式 Stage A 的生成器、分數與 source snapshot 未覆寫。

正式完整批次已於台灣時間 2026-10-03 01:43:26 啟動：`results/v2_full_suite_20261003`；共 284 jobs／7,240 evaluations／28,960 candidate evaluations（實際 fits 因等價性引用較少）。先跑 B 五軸，其他實驗自動接續；完整批次不承諾一晚完成。方法、續跑與查看結果的入口見 [SIMULATION_SUITE.md](SIMULATION_SUITE.md)。

## 2026-10-03 中斷與恢復

查核時原批次 status 為 failed：先有 Windows status.json 寫入遭拒，再有 supervisor failure timestamp 重複造成 TypeError。Stage B 五軸、四個 sample sizes、400-epoch sensitivity 已完成；保存了 1,072 次模型評估。

修正唯一暫存檔／有界 I/O retry 與 failure event timestamp；新增回復來源／computation modules 與 AST equality／checkpoint checksum guards。71 項測試通過，cache-copy 調整後 6 項相關測試通過。於台灣時間 11:03 啟動 `results/v2_full_suite_recovered_20261003`，完整保存舊批次，驗證並沿用 1,072 份權重／分數；11:09 已新增一次模型評估，兩個 workers 持續執行。此修復版本 source SHA256：`d2fb3976497a58b67d4dfe139f52ea6c645149efc1652d05a534199ee2dcf5d2`。

結果與非單調 dynamics curve 的限制見 [進度摘要](results/v2_full_suite_recovered_20261003/PROGRESS_20261003.md)。不把 partial C/D maps 稱為完整研究成果。
