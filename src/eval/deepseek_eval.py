from src.model_loader.base_pipeline import load_model_pipeline
from src.model_loader.four_bit_quantized import load_quantized_model, get_quantized_response
import json
import os
from dotenv import load_dotenv
load_dotenv()

#*******************************
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#*******************************

def deepseek_chat():
    """
    probably self explantory
    """
    
    count = -1
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com") 
    
    with open(r"C:\Users\allan\nvim\tinyMath\TinyMathLLM\src\eval\base_llama\answer.json", "r", errors = "ignore") as model_output:
        with open(r"C:\Users\allan\nvim\tinyMath\TinyMathLLM\src\deepseek_judge\answer\deepseek_answer.json", "a") as deepseek_eval:
            for line in model_output:
                temp = {}
                count += 1
                if count % 2 == 0:
                    data = json.loads(line.strip())
                    question = data["input"][1]
                    answer = data["output"]
                
                    messages = [
                        {
                            "role": "system",
                            "content": """Provided with the the following question and answer, first, identify what kind of question it is from this list: ["Addition", "Multiplication", "Derivation", "Integration", "MatrixMultiplication", "Determinant"]. Then respond if answer is correct. Your reponse should look like 'Addition: NO'. Here is the question: {}. Answer: {}. DO NOT PROVIDE AN EXPLANATION. YOUR ANSWER SHOULD LOOK ONLY BE 2 WORDS.""".format(question, answer)
                        }
                    ]

                    messages.append({"role": "user", "content": "Question: {}. Answer: {}".format(question, answer)})
                    response = client.chat.completions.create(
                         model="deepseek-chat",
                         messages=messages,
                         max_tokens=1024,
                         temperature=0.7,
                         stream=False
                    )
                    bot_reply = response.choices[0].message.content
                    print(bot_reply, count)
                    temp["Question"] = question["content"]
                    temp["Judgement"] = bot_reply
                    json_line = json.dumps(temp)
                    deepseek_eval.write(json_line + "\n")
     
deepseek_chat()





