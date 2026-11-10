# 第二版：資料機制與模型相對優勢

完整背景佇列、固定預算、收斂檢查與明早的檔案入口見 [SIMULATION_SUITE.md](SIMULATION_SUITE.md)。原單階段 CLI 保留。

這版研究的是「哪些可量測特徵能解釋模型的相對優勢」，主要證據為 performance-gap response curves。第一版程式、網站與 `results/overnight_20260930` 的封存結果保留；第二版在獨立套件 `src/gnn_mechanisms` 執行，不混合兩版分數。

## 研究計畫的採納與必要修正

採用五個主模型：MLP、Static-GCN、Static-GAT、Dynamic-GCN、Dynamic-GAT。Dynamic 指每個預測起點使用當下拓撲；五個模型都只讀當下八維特徵，不讀反應歷史、不使用節點 ID，也不加入 recurrent memory。選擇 nonlinear node encoder、單層 aggregation、共同 self branch 與 nonlinear readout；GCN 在 aggregation 前已能處理 nonlinear neighbor content。這是指定架構的機制比較，不能外推成所有 GNN 架構的排名。

Static-GAT 固定的是 topology，attention weights 仍隨特徵改變。文獻的「static attention」指 neighbor ranking 不依賴 query；它與本研究的 Static/Dynamic graph 是不同分類。

1. 注意力熵需帶負號：`H_att = -sum(alpha log alpha)/log(neighbor_count)`，`R=1-H_att`。僅有 self-loop 的孤立節點定義 R=0。
2. 依 **目標時間** 切分。預測 Y(t+1) 時，N=500、T=40 使用目標時間 2–24 訓練、25–32 驗證、33–40 測試；training origins 為 1–23。時間 40 沒有研究期間內的下一步標籤，所以不納入預測。Smoothing 實驗為同時點 denoising，切分為 1–24、25–32、33–40，結果不與 next-step regression 混成一條曲線。
3. lambda 在 training-standardized u、v 下控制兩個成分的**設計變異配比**，不包含 response noise；有限样本 u/v covariance 可能非零，會記錄，不將 lambda 宣稱為總反應的精確解釋變異。
4. tilde-neighborhood 包含自身，故 lambda=1 時仍有一部分 graph-feature signal 可由自身觀測；「self / graph feature blocks 獨立」不能消除這個直接來源。這是與 GCN self-loop 對齊的明確設計選擇。
5. 純圖雜訊下 D_pred 可以為負值：noisy dynamic graph 可能比 consensus static graph 更差。檢驗「沒有正的 dynamic predictive gain」，不強迫 D_pred 等於 0。
6. delta、eta 指替換原圖邊的比例。每次 double-edge swap 更換兩條原邊；使用最接近的偶數邊數，保留 degree、density、connectivity，另外記錄實際 turnover。Jaccard dynamicity 與 turnover 是不同尺度。
7. Smoothness 實驗的 z_tau 預設以 training moments 標準化，再加固定 observation/response noise；避免 tau 同時大量改變 SNR。原始 resolvent 可用 `smooth_normalize=false` 作 robustness。
8. 主研究直接觀察生成圖的二元 topology，再視情境加入觀测雜訊；**不從 response 估計 topology**。這先隔離 aggregation/topology 機制，資料建圖誤差屬後續外部轉譯問題。生成用的實數權重不提供給模型。

研究假說保留原計畫的方向性，未將任何 winner / curve direction 寫成驗收測試。測試只驗證生成機制、數學不變量、模型實作與資料邊界。

## 生成器

- 正式預設：N=500、T=40、p=8、d=10、phi=0.6、60/20/20 split。
- 每個 feature 由 stationary AR(1) 生成，初值 N(0,1)，innovation scale=sqrt(1-phi^2)。self 四維與 graph 四維為獨立亂數；圖、漂移、觀測雜訊、features、response noise、smooth latent 各有獨立亂數流。
- 固定 `f(x)=x1+0.5*x2^2+sin(x3)+0.5*x3*x4`；`g(x)=x1+0.5*x2^2+0.5*sin(x3)+0.3*x4`。原訊息部分數學符號排版不完整，這裡明確採用上述加號。
- source relevance 以固定單位向量 b=(1,-0.5,0.25,0.5)/norm 產生；weights=softmax(kappa*b'X_graph)，只在當下 true neighbors 加自身上計算。kappa=0 為 uniform aggregation。
- u、v 各用 training-target 對應的 feature origins 校準為零均值、單位變異；Y_next=sqrt(1-lambda)*u+sqrt(lambda)*v+sigma*epsilon。
- True drift 對上一期 true graph 重接；noise 每一期獨立對當下 true graph 重接，不讓 noisy observations 反過來影響 targets。
- Static consensus 使用 graph times <=24 的完整訓練期 snapshots，按 edge frequency 選 M 條邊，M=動態圖平均邊數；平手以獨立固定亂數處理，不再使用 >=0.5 多數決刪到接近空圖。相同 edge count 不保證 consensus graph 仍 d-regular，會保留這個限制。
- Isolated smoothing 用 z_tau=(I+tau*L)^(-1)z0；只讓 feature 1 是 noisy z，其餘七維為獨立 distractors。Stage C/D 的 tau 使用另一個明確延伸：平滑 graph feature block，self block 保持獨立；不能把兩種 tau 實驗當成同一個生成器。

## 可觀測指標與 oracle 分離

S_X、S_G、D_pred 使用 training period 內的 expanding chronological folds。每個 fold 只在較早時間訓練固定 probe，在後續 training-period 時間評估；不使用主研究 validation/test。所有 fold 的 out-of-fold predictions 合併計算 R2：

- Probe basis 為自身八維的 quadratic terms 與 sin terms，固定 Ridge(alpha=10)，design standardization 只 fit fold-training rows。
- Neighbor summary 為包含 self-loop 的 uniform mean of the same basis。
- S_X = R2(feature probe)；S_G = R2(feature+dynamic summary)-R2(feature probe)。
- 負對照用固定 node permutation 重標 observed graph，保留圖的 density/degree 與 temporal structure，再計算 S_G_shuffled。
- D_pred = R2(dynamic-summary probe)-R2(static-summary probe)；每個 fold 的 static consensus 只使用該 fold 的 training snapshots。
- S_G 是**固定 uniform-summary probe 可擷取資訊的下界診斷**，不是所有可能 graph predictors 的資訊量。高 kappa 時它可能低估 selective information；curve 不成立時必須把這個限制納入討論。
- R2 允許負值；S_G 是 R2 差值，也不限制在 0–1。小樣本的負值／大幅差值應連同 probe errors 與 seed variability 解讀，不能裁切成理想結果。
- H = 1-E(Y)/E_perm[E(pi(Y))]，E 為 undirected edge squared differences。Permutation expectation 用解析式 `2*N/(N-1)*Var(Y)*edge_count` 精確計算；H 可為負數，不裁切成正值。
- D_struct 是相鄰 observed snapshots 的平均 Jaccard distance。
- R_true 只放在 oracle/mechanism analysis；R_hat 由固定、低容量的 source-feature edge-gating probe，用 training labels 估計。它是一個估計器，不預設一定找回真權重；輸出 signed estimation error。
- S_G/R_hat/D_pred 都依賴指定 probe 的能力，不能稱為完全 model-independent。Phase maps 預設使用 R_hat；adaptive mechanism curve 另顯示 oracle R_true，清楚標示。
- Classification robustness 的 diagnostics 使用 binary target 的 R2 probes；與 regression maps 分開。可再另增 probabilistic classification probes，不把不同 score scales 混合。

## 執行順序與指令

以下指令在專案根目錄的 PowerShell 執行。第一步測試生成器／邊界，第二步驗證小規模全流程。Smoke 的兩個 seeds 與八輪訓練只驗證程式，不作研究結論。

```powershell
& algorithm\.venv\Scripts\python.exe -m unittest discover -s algorithm/tests -p test_mechanisms_v2.py -v
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --config algorithm/configs/mechanism_v2_smoke.json --output algorithm/results/v2_smoke
& algorithm\.venv\Scripts\python.exe algorithm/check_mechanisms.py --output algorithm/results/v2_generator_review.json
```

每階段先 dry-run 檢查工作量，再指定獨立輸出資料夾。預設正式設定只執行 Stage A，不自動啟動整個大型研究：

```powershell
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --stages A --dry-run
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --stages A --output algorithm/results/v2_stage_A
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --stages B --output algorithm/results/v2_stage_B
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --stages C --output algorithm/results/v2_stage_C
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --stages D --output algorithm/results/v2_stage_D
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py simulate --stages sample_size --output algorithm/results/v2_sample_size
```

Stage A 六個極端生成機制；Stage B 28 個 one-axis references；Stage C 56 個 interaction references；Stage D 預設 200 個 Latin-hypercube conditions。重複的 condition/seed 在同一批次只訓練一次，但各實驗保留自己的 reference；不累計總勝場。stage 啟動順序由研究者依機制檢查決定，不用「假說成立」作自動通關門檻。

Robustness 包含 heterogeneous-degree、SBM、獨立幾何 kNN、noise、classification 與 query-dependent relevance。把設定複製為新 JSON 並設 `gatv2=true`，才額外訓練 Static/Dynamic-GATv2；五個主模型始終保留。每個主／額外模型使用相同 candidate grid。

Stage A 正式設定為 30 datasets、150 模型、600 candidate fits。Stage D 預設為 1,000 datasets、5,000 模型、20,000 candidate fits；可先修改 sampled condition count，不能將一次小規模驗證稱為完整研究。

## 訓練、公平性與配對推論

五模型共享 preprocessing、chronological splits、同樣的 loss、candidate grid、early-stopping tolerance、patience、max epochs。正式 hidden widths=[32,64]、learning rates=[0.001,0.003]，每模型四候選；regression 用 MSE，classification 用 BCE。validation 選超參數與 checkpoint，test 在選定後才計分。Snapshot batches 保留整張圖的所有 nodes；shuffle training batches 不改變 split。

實作使用 padded sparse neighborhoods；避免 GAT 為每個快照建立 N*N*hidden 的 dense pair tensor。GCN 使用 symmetric normalized adjacency；vanilla GAT 使用 additive masked attention；optional GATv2 提供 query-dependent ranking。MLP 同樣有 nonlinear encoder/readout 與合理候選容量。

所有 comparison 使用相同 dataset seed：

- 四條 GNN-minus-MLP gaps，各自報告，不以 test 最佳 GNN 取最大值。
- Static / Dynamic 的 GAT-minus-GCN。
- GCN / GAT 的 Dynamic-minus-Static。
- Interaction=(GAT_D-GAT_S)-(GCN_D-GCN_S)。

主要 score 是 regression R2、classification AP；另記錄 RMSE/MAE 或 AUROC/Brier/log loss。Bootstrap 以**完整 independent dataset seeds**為配對單位，不能把相關 nodes/times 當成獨立重複。P(delta>0) 指「種子中 delta>0 的比例」，不是 Bayesian posterior probability 或 p-value。少於五個 seeds 標為 inference_ready=false。

## 產出、續跑與網站接入

- `manifest.json`：設定、source SHA256、套件版本與工作量；`source/` 保存可檢視的源碼快照。
- `status.json`：目前進度；`run.lock` 防止同時執行；`complete.json` 是完成標記。
- `datasets/<condition>/seed_<s>/generator.json`：實際 turnover、校準 moments、oracle 量、origin/target/split times。
- `diagnostics.json`：training-only diagnostics 與內部 chronological folds。
- 每個模型保存 `weights.pt`、`predictions.npz`、`training.json`、`result.json`；completion marker 最後寫入。
- `metrics.csv`、`paired_contrasts.csv`、`response_curves.csv`、`phase_points.csv`。
- `figures/`：Stage B response curves；Stage C/D 用 observable diagnostics 畫 continuous performance-gap phase maps，不硬切 winner regions。
- `website_results_v2.json`：第二版獨立匯出介面；不自動覆蓋網站第一版結果。

相同設定與相同源碼可用相同指令續跑完整 checkpoints。設定或源碼改變時，程式要求新目錄，避免混批。正常中斷會清掉 lock；若程序被系統強制結束，先確認該 PID 已停止，再移除殘留 lock。Report-only 指令可重新匯出已保存分數，不重新訓練。

## 石門水庫外部驗證介面

此版本提供凍結 expectation 與事後 comparison 的程式；**不表示已完成實際高濁度驗證**。現有第一版資料的 validation/test 無 >1000 NTU 正例等缺口仍需處理。主 synthetic study 是 snapshot 一步 regression；真實 1/3/6 小時事件窗口須另外驗證時間尺度、labels、missingness 和 diagnostics 的可移轉性。

外部 input 為**只包含 training period**的 NPZ：

- features=[time,node,feature]，target=[time,node]，graphs=[time,node,node] 為二元 undirected snapshots。
- target_times 必須已對齊預測目標時間；下一步 target 要與其 feature origin 對齊，不能把同時點資料誤當 next-step。
- metadata 是 JSON scalar，至少包含 training_only=true、training_cutoff、dataset_id、target alignment/source provenance。程式拒絕 cutoff 以後的 targets，不替使用者驗證上游資料的發布延遲。

```powershell
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py freeze-expectation --panel TRAIN_ONLY.npz --synthetic algorithm/results/v2_stage_D --output algorithm/results/shihmen_expectation.json
& algorithm\.venv\Scripts\python.exe algorithm/run_mechanisms.py validate-real --expectation algorithm/results/shihmen_expectation.json --heldout REAL_HELDOUT.json --output algorithm/results/shihmen_validation.json
```

Expectation 文件不可覆寫，拒絕使用 smoke 結果作外部預註冊；包含來源 checksums、凍結時間、training diagnostics、diagnostic/node-count support checks，與附近 synthetic conditions 的相對模型差異。不預先指定 winner，也不把附近情境差異範圍稱為 prediction interval。Heldout JSON 必須引用 expectation SHA256、相同 dataset_id、較晚的 evaluated_at，以及五個模型各自的 test_r2。這是 regression translation 的初始介面；classification / 1-3-6h 外部機制圖需匹配 task 後另外擴充。

## 理論來源

- [GCN：一階局部 spectral convolution approximation](https://arxiv.org/abs/1609.02907)
- [GAT：learned masked neighbor attention](https://arxiv.org/abs/1710.10903)
- [Vanilla GAT ranking limitation 與 GATv2](https://arxiv.org/abs/2105.14491)
- [GCN 與 Laplacian smoothing 的關係](https://arxiv.org/abs/1801.07606)

上述文獻支持架構與機制的關係，不保證任何模型在本研究的生成器、樣本數或訓練预算下必然獲勝。
