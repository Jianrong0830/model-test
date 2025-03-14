# ibo.ai大型語言模型測試流程

## 一、測試模型及資源需求

本測試流程旨在評估多個大型語言模型的性能和準確性，所需的測試模型及硬體資源如下：

### 1. NLKE 測試

本測試將評估模型對於具體知識點的準確檢索能力，使用事先準備的測試資料進行。

#### 測試資料
測試資料為公司資訊和部門相關的知識點，具體如下：

| 標題     | 知識點                         |
|----------|--------------------------------|
| 公司名稱 | 大同世界科技股份有限公司       |
| 公司地址 | 台北市中山區中山北路三段22號  |
| 部門名稱 | G5-AI創新服務處               |
| 部門工作 | AI客服機器人                  |
| 部門主管 | Hector                           |

#### 問題與正確標籤範例
- 測試問題和標籤（filter）樣例如下，每個知識點設計至少10個問題：
  - 你在哪間公司上班？ → 正確標籤: 1
  - 公司位於台北的哪個區域？ → 正確標籤: 2
  - 你在哪個部門？ → 正確標籤: 3
  - G5-AI創新服務處的主要工作是？ → 正確標籤: 4
  - 請問部門主管是誰？ → 正確標籤: 5

#### 測試步驟
1. 利用和ibo相同的 `requests` 格式請模型回答問題，期待回傳 JSON 格式的資料，如： `{'filter': '4', 'text': ' AI客服機器人', 'quickreply': None}`
2. 檢查每個問題的回答是否包含正確的標籤（filter），並計算模型的正確率：
   - **正確率** = 正確回答題數 / 總題數
3. 若模型回答的 JSON 格式不正確，也進行記錄以計算格式錯誤率：
   - **格式錯誤率** = JSON 格式錯誤的回答數 / 總題數
4. 根據正確率和格式錯誤率，評估模型的表現。

## 三、結論與模型效果評估

最終的模型效果將根據以下幾個指標進行綜合評估：
1. **NLKE 測試的正確率**：準確回應指定知識點的能力。
2. **NLKE 測試的格式錯誤率**：JSON 格式輸出正確性。

綜合這些指標數據，以判定各模型在處理結構化知識問答和上下文理解能力方面的優劣程度。

### 參考程式碼
測試程式碼與資源請參見 [Model-Test](http://172.31.10.92:9091/Ryan-JR.Chen/model-test)。

## 二、測試結果
| 模型                                | 正確率 | 格式錯誤率 |
|-------------------------------------|--------|------------|
| qwen2.5-72b-inst-32k               | 98%    | 0%         |
| ffm-mixtral-8x7b-32k-instruct      | 88%    | 2%         |
| gpt-4o                             | 84%    | 0%         |
| gpt-4                              | 80%    | 0%         |
| meta-llama3.3-70b-inst-32k         | 78%    | 0%         |
| deepseek-r1-distill-llama-70b      | 76%    | 0%         |
| llama3-ffm-70b-chat                | 70%    | 0%         |
| llama3.1-ffm-70b-32k-chat          | 68%    | 0%         |
| llama3.2-ffm-11b-v-32k-chat        | 62%    | 0%         |
| gpt-35-turbo-16k                   | 62%    | 0%         |
| llama3-ffm-8b-chat                 | 60%    | 0%         |
| mistral-7b                         | 50%    | 34%        |
| gpt-4o-mini                        | 44%    | 0%         |
| Taiwan-LLM-13B-v2.0-chat           | 26%    | 8%         |
| gemma-2b                           | 12%    | 2%         |
| Taiwan-LLM-7B-v2.1-chat            | 0%     | 46%        |
| Breeze-7B-Instruct-v1_0            | 0%     | 100%       |
| TAIDE-LX-7B-Chat                   | 0%     | 100%       |

**詳細結果請參考[Model-Test](http://172.31.10.92:9091/Ryan-JR.Chen/model-test)的output資料夾**