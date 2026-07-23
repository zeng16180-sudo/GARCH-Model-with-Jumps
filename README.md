# GARCH Model with Jumps
## 簡介
本專案使用以下三種跳躍模型，估計模型參數。可用於分析跳躍強度（jump intensity）的動態變化，或更進一步做報酬率的預測。
- ARJI-GARCH Model(Autoregressive Jump Intensity Model): 以自迴歸方式描述跳躍強度的時間變化。
- SEJI-GARCH Model(Self-Exciting jump intensity Model): 考慮跳躍事件可能提高後續跳躍發生機率。
- HJI-GARCH Model(Hawkes jump intensity Model): 使用 Hawkes process 描述跳躍事件的自我激發特性。
## 環境需求
- Python 3.x
- NumPy
- Pandas
- SciPy
- Matplotlib

安裝套件：

```bash
pip install numpy pandas scipy matplotlib
```
## 檔案說明
- `ARJI.py`：ARJI-GARCH Model參數估計
- `SEJI.py`：SEJI-GARCH Model參數估計
- `HJI.py`：HJI-GARCH Model參數估計
- `data/`：資料檔案（報酬率檔案）
- `output/`：分析結果與圖表(参數估計結果)

## 使用方式
1. 將報酬率資料放入 data/ 資料夾。
2. 執行想使用的模型：
`python ARJI.py`
`python SEJI.py`
`python HJI.py`
3. 參數估計結果與圖表會儲存在 output/ 資料夾。

## Reference
Chan, W. H., & Maheu, J. M. (2002). Conditional jump dynamics in stock market returns. *Journal of Business & Economic Statistics, 20*(3), 377–389. https://doi.org/10.1198/073500102288618513

Zhang, L., Chen, Y., & Bouri, E. (2024). Time-varying jump intensity and volatility forecasting of crude oil returns. *Energy Economics, 129*, Article 107236. https://doi.org/10.1016/j.eneco.2023.107236
