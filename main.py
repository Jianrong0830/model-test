import pandas as pd
import time
import json
import re
import logging
import os

from llm import LLM

class NLKETest:
    def __init__(self, model):
        self.model = model
        self.llm = LLM(model)
        self.km = pd.read_excel('NLKE參考資料.xlsx')
        # 用於儲存所有模型的結果
        self.all_results = []

    def test_model(self, prompt, callback=None):
        prompt += "\n\n###\n參考資料：\n請幫我依照以下文檔內容分類依據判斷問題屬於何者，若都不屬於、屬於多個或是比較項目請在filter填入0\n\n文檔內容：\n"
        history = [
            {"role": "user", "content": self.km.loc[0, '標題']},
            {"role": "assistant", "content": f'{{"filter":"1","text":"{self.km.loc[0, "知識點"]}","quickreply":null}}'},
            {"role": "user", "content": self.km.loc[1, '標題']},
            {"role": "assistant", "content": f'{{"filter":"2","text":"{self.km.loc[1, "知識點"]}","quickreply":null}}'},
            {"role": "user", "content": self.km.loc[2, '標題']},
            {"role": "assistant", "content": f'{{"filter":"3","text":"{self.km.loc[2, "知識點"]}","quickreply":null}}'},
            {"role": "user", "content": self.km.loc[3, '標題']},
            {"role": "assistant", "content": f'{{"filter":"4","text":"{self.km.loc[3, "知識點"]}","quickreply":null}}'},
            {"role": "user", "content": self.km.loc[4, '標題']},
            {"role": "assistant", "content": f'{{"filter":"5","text":"{self.km.loc[4, "知識點"]}","quickreply":null}}'},
        ]
        correct = 0
        json_errors = 0

        test_question = pd.read_excel('NLKE測試題目.xlsx')
        total_questions = len(test_question)
        model_results = []

        if callback:
            callback("開始測試", 0, total_questions)

        for index, row in test_question.iterrows():
            user_input = row['問題']
            correct_filter = row['正確filter']
            ai_filter = 0  # 預設值
            ai_response_text = ""  # AI實際回答的文本
            answer_correctness = "錯誤"  # 回答是否正確
            format_correctness = "正確"  # 格式是否正確

            try:
                # 取得原始回應
                response = self.llm.generate_response(user_input=user_input, history=history, prompt=prompt)
                
                # 移除 <think> 與 </think> 之間的所有內容
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                
                ai_response_text = response.strip()
                # 移除不必要的字串，例如 'json' 與三個反引號
                response = response.replace('json', '').replace('```', '')

                # 嘗試將回應轉換成 JSON 物件以擷取 filter
                try:
                    response_json = json.loads(response)
                    ai_filter_str = response_json.get('filter', '0')
                    ai_filter = int(ai_filter_str)
                except json.JSONDecodeError:
                    format_correctness = "錯誤"
                    json_errors += 1
                    ai_filter = 0  # 若 JSON 格式錯誤，則無法取得 filter
                except (ValueError, TypeError):
                    ai_filter = 0  # 無法擷取 filter

                if ai_filter == correct_filter:
                    correct += 1
                    answer_correctness = "正確"
                else:
                    answer_correctness = "錯誤"

            except Exception as e:
                logging.error(f"Error processing question '{user_input}': {e}")
                continue

            # 儲存結果
            model_results.append({
                "問題": user_input,
                "正確filter": correct_filter,
                "AI回答的Filter": ai_filter,
                "AI回答的文本": ai_response_text,
                "回答": answer_correctness,
                "格式": format_correctness
            })
            print(f"問題：{user_input}, 回答：{ai_response_text}")

        # 計算準確率和格式錯誤率
        accuracy = correct / total_questions if total_questions > 0 else 0
        error_rate = json_errors / total_questions if total_questions > 0 else 0
        print(f"accuracy: {accuracy}, error rate: {error_rate}")

        return {"results": model_results, "accuracy": accuracy, "error_rate": error_rate}

def run_test_for_model(model, prompt, output_dir="output"):
    """
    測試指定的模型並將結果匯出為 Excel 檔案
    """
    # 初始化 NLKETest
    nlt = NLKETest(model)
    
    # 執行測試
    result = nlt.test_model(prompt=prompt)
    
    # 若 output 目錄不存在則建立
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f'{model.replace(':', '-')}_results.xlsx')
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # 詳細結果寫入 "Results" sheet
        df_results = pd.DataFrame(result["results"])
        df_results.to_excel(writer, sheet_name='Results', index=False)
        # 統計資料寫入 "Summary" sheet
        df_summary = pd.DataFrame({
            'accuracy': [result["accuracy"]],
            'error_rate': [result["error_rate"]]
        })
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f"模型 {model} 的結果已匯出至 {output_file}")

if __name__ == "__main__":
    # 讀取 Prompt 內容
    with open('NLKE測試範例Prompt.md', 'r', encoding='utf-8') as file:
        prompt = file.read()
    
    # 定義要測試的模型列表
    models_to_test = [
        'ffm-mixtral-8x7b-32k-instruct',
    ]
    
    # 對每個模型進行測試
    for model in models_to_test:
        print(f'開始評測 {model}')
        run_test_for_model(model, prompt)
