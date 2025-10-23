from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import json
from datetime import datetime
from utils import get_store
from deepseek_api import get_api_result
from tongagents.agents.llm.base import ModelConfig
from tongagents.agents.llm_agent.react_agent import (
    LLMRunContext,
    ReactAgent,
    ReactAgentSetting,
)
from tongagents.tools.tool_manager import ToolManager
from tongagents.agents.llm.messages import (
    ModelOutputMessage,
    ModelTextMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
import copy

SecretId = os.getenv('SECRET_ID')
SecretKey = os.getenv('SECRET_KEY')
api_key = os.getenv('API_KEY')

app = Flask(__name__, static_folder='static')
CORS(app)  # Allow cross-origin requests

store_list = None
ToolManager.initialize_from_remote(["10.1.53.30:8000"])
# Configure the table-tennis agent system prompt; focus on table tennis and match rules
agent_settings = ReactAgentSetting(
    # New system prompt instructing the model to act as a table-tennis rules assistant
    system_prompt="你是乒乓球规则智能体，专注于提供关于乒乓球运动及其相关比赛规则的信息。你能够解答与乒乓球的规则、历史、技术以及比赛制度等有关的问题。",
    llm_config=ModelConfig(
        model_name="deepseek-v3",
        url="https://api.lkeap.cloud.tencent.com/v1",
        api_key=api_key,
        stream=True
    ),
    tool_identifier_list=["web_search"],
)
# Create and configure the agent
agent = ReactAgent(
    dep_context=LLMRunContext(),
    agent_settings=agent_settings,
)

@app.before_request
def init_once():
    global store_list
    global agent
    store_list = [
        {
            "store": get_store('test_hybrid', 'local_db/tongagents_milvus_multi_0609.db'),
            "threshold": 0.5,
            "max_ratio": 0.0,
            "recall_num": 10,
            # Update the store name to reflect table-tennis content
            "store_name": "乒乓球知识库1"
        },
        {
            "store": get_store('tong_knowledge_offline_store', 'local_db/bigai_milvus_0702_offline_2.db'),
            "threshold": 0.48,
            "  ": 0.4,
            "recall_num": 10,
            # Update the store name to reflect table-tennis content
            "store_name": "乒乓球知识库2"
        },
    ]

def get_avg_score(results):
    return sum([sample.score for sample in results]) / len(results)

def get_max_score(results):
    return max([sample.score for sample in results])

def judge_recall_valid(results, threshold, max_ratio):
    return get_avg_score(results) * max_ratio + get_max_score(results) * (1 - max_ratio) > threshold

def pre_agent_memory(system_prompt, context_history):
    agent.run_context.deps.memory.store = []
    msgs = [SystemPromptMessage(content=system_prompt)]
    for i in range(len(context_history)):
        msgs.append(UserPromptMessage(context_history[i]['Content']))
        msgs.append(ModelOutputMessage(model_text_message=ModelTextMessage(content=context_history[i]['Content'])))
    for msg in msgs:
        agent.run_context.deps.memory.add_chunk(msg)
    
def query_rewrite_module(current_query, chat_history):
    if len(chat_history) == 0:
        # When there is no chat history, guide query rewriting with the table-tennis prompt
        rewrite_query = f"你是乒乓球规则智能体，专注于回答乒乓球及其相关比赛规则的问题。\n请根据下列用户输入，将其补全为一个完整且清晰的问句：\n\n{current_query}\n\n注意：\n1. 如果当前用户的问题已经完整，无需改写，请原样输出。\n2. 对于用户的拼写错误，也请保持原样。\n3. 只输出最终结果，**不要添加任何说明或解释**。\n\n下面请输出结果：\n"
    else:
        rewrite_query = f"你是一个对话问题改写助手，擅长理解上下文中的指代关系与语义依赖。你的任务是根据用户当前的问题和对话历史，进行指代消解，补全为清晰、完整、无歧义的问题句子。\n\n请根据下面的对话历史，对当前用户的问题进行指代消解，并将其补全成一个完整句子。\n\n【对话历史】：\n{chat_history}\n\n【当前用户的问题】：\n{current_query}\n\n请补全当前用户的问题，使其不含指代词或歧义，并且不改变问题的语义。\n\n注意：\n1. 如果当前用户的问题是完整的，无需改写，请原样输出。\n2. 如果当前用户的提出了与对话历史无关的新问题，也请不要改写，直接原样输出。\n3. 对于用户的拼写错误，也请保持原样。\n4. 只输出最终结果，**不要添加任何说明或解释**。\n\n下面请输出结果：\n"
    context = [{'Role': 'system', 'Content': ''}, {'Role': 'user', 'Content': rewrite_query}]
    return ''.join([x for x in get_api_result(context, secret_id=SecretId, secret_key=SecretKey)]) if len(chat_history) > 0 else current_query

def query_judge(query, keywords=[]):
    judge_time = 0
    while True:
        judge_query = f"你是一个query判断助手，你需要判断用户输入的query是否与{'、'.join(keywords)}相关，并直接输出“是”或者“否”，以下是需要判断的query：\n\n{query}\n\n下面请输出结果："
        context = [{'Role': 'system', 'Content': ''}, {'Role': 'user', 'Content': judge_query}]
        judge_time += 1 ##
        result = ''.join([x for x in get_api_result(context, secret_id=SecretId, secret_key=SecretKey)])
        if result in ['是', '否'] or judge_time >= 3:
            break
    return result if result in ['是', '否'] else '否'

def get_tongagents_result(user_input, recall_content, context_history):
    # Build the table-tennis agent system prompt with current time info
    system_prompt = f'你是乒乓球规则智能体，专注于解答乒乓球运动及其比赛规则相关的问题。现在是{datetime.now().strftime("北京时间%Y年%m月%d日%H时%M分")}。'
    prompt = f'以下是多个不同来源的知识内容，***以上内容可能与问题相关，也有可能与问题无关***。内容之间用“########”分隔：\n{recall_content}\n你需要根据以上内容，并结合历史对话，以准确简洁的方式回答用户的问题；如果答案涉及多个方面时，也可以分点回答。用户的问题可能没有合适的答案，如果没有合适答案，请如实告知用户。用户的问题是：{user_input}\n下面请输出答案：'
    context = [{'Role': 'system', 'Content': system_prompt}] + context_history + [{'Role': 'user', 'Content': prompt}]
    print(system_prompt)
    return get_api_result(context, secret_id=SecretId, secret_key=SecretKey, model='deepseek-v3')
    #n_rounds = 0
    #while True:
    #    response = ''.join([x for x in get_api_result(context, secret_id=SecretId, secret_key=SecretKey, model='deepseek-v3')])
    #    n_rounds += 1
    #    if '【最终输出】' in response or n_rounds >= 3:
    #        break
    return {"system_prompt": system_prompt, "response": response}

def get_general_result(user_input, context_history):
    # Build the table-tennis agent system prompt, describe its ability, and include the current time
    system_prompt = f'你是乒乓球规则智能体，专注于解答乒乓球运动及其比赛规则相关的问题。\n你能够提供乒乓球的规则、历史、技术以及比赛制度等信息。\n现在是{datetime.now().strftime("北京时间%Y年%m月%d日%H时%M分")}。'
    now_context_history = copy.deepcopy(context_history)
    pre_agent_memory(system_prompt, now_context_history)
    print(system_prompt)
    for r in agent.stream(user_input):
        #if getattr(r, "is_final_response", False):
        yield r.content

def get_result(user_input, context_history, store_list, max_recall_num=12, threshold=0.5):
    #print(f"user_input: {user_input}")
    #print(f"context history: {context_history}")
    rewrite_count = 0
    while True:
        rewrite_input = query_rewrite_module(user_input, context_history)
        rewrite_count += 1 
        if rewrite_count >= 5 or '改写' in user_input or '改写' not in rewrite_input:
            break
    # Keywords used to judge whether the query is related to table tennis
    key_word_list = ['乒乓球', 'pingpong', '乒乓球比赛', '比赛规则', '国际乒联', '乒乓', '球拍']
    query_judge_result = query_judge(rewrite_input, keywords=key_word_list)
    print(rewrite_input)
    print('*'*20)
    if query_judge_result == '是':
        recall_results = []
        for store in store_list:
            store_recall_outs = ['store'].search(rewrite_input, limit=store['recall_num'])
            print(store_recall_outs)
           
            if judge_recall_valid(store_recall_outs, store['threshold'], store['max_ratio']):
                recall_results += store_recall_outs
        #print(f"Recall results: {recall_results}")
        if len(recall_results) > 0:
            recall_results = sorted(recall_results, key=lambda x: -x.score)[:max_recall_num]
            recall_content = '\n\n########\n\n'.join(list(set([out.document.content.strip() for out in recall_results])))
            resp = get_tongagents_result(rewrite_input, recall_content, context_history)
        else:
            recall_content = ''
            resp = get_general_result(rewrite_input, context_history)
    else:
        recall_results = []
        recall_content = ''
        resp = get_general_result(rewrite_input, context_history)

    return rewrite_input, resp


# Assume this is your already-implemented LLM function
def your_model_function(user_input, history=None, max_round=10, recall_num=10, threshold=0.5):
    # You can use the `history` parameter to provide conversation context
    # If your model does not need history, you can ignore this parameter
    
    # Simple example
    #if history and len(history) > 2:
    #    response = f"Based on the previous conversation, my reply is: {user_input}"
    #else:
    #    response = f"This is a reply from the model: {user_input}"
    history = history[-max_round*2-1:-1]
    response = get_result(user_input, history, store_list, recall_num, threshold)#['response']
    return response

@app.route('/api/chat_with_api', methods=['POST'])
def chat_with_api():
    data = request.json
    user_input = data.get('message', '')
    history = data.get('history', [])  # Get chat history
    
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    # Call your model function with the chat history
    return jsonify(your_model_function(user_input, history))

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    history = data.get('history', [])  # Get chat history
    
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    # Call your model function with the chat history
    response = your_model_function(user_input, history)['response']#.split('【最终输出】')[-1]
    
    return jsonify({"response": response})

@app.route('/stream', methods=['POST'])
def stream():
    data = request.json
    user_input = data.get('message', '')
    history = data.get('history', [])

    def generate():
        max_attempts = 3
        default_max_round = 10
        for attempt in range(max_attempts + 1):
            prefix_buffer = []
            started = False
            char_threshold = 30
            user_input_norm = user_input.strip()[:char_threshold]
            rewrite_input, generator = your_model_function(user_input, history, max_round=default_max_round - attempt)
            rewrite_input_norm = rewrite_input[:char_threshold]

            for text_chunk in generator:
                if text_chunk is None:
                    continue
                prefix_buffer.append(text_chunk)
                current_output = ''.join(prefix_buffer).strip()

                if not started:
                    if len(current_output) >= char_threshold:
                        if (current_output.startswith(user_input_norm) or current_output.startswith(rewrite_input_norm)) and attempt < max_attempts and len(user_input) >= 5:
                            break
                        else:
                            for cached in prefix_buffer:
                                yield f"data: {json.dumps({'text': cached})}\n\n"
                            started = True
                    continue

                yield f"data: {json.dumps({'text': text_chunk})}\n\n"

            if started:
                break

        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# Serve static files (frontend page)
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Optional: add an API endpoint to clear server-side chat history (if you maintain it)
@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    # If you maintain user sessions on the server, you can clear them here
    # For example, if using Flask sessions:
    # session['chat_history'] = []
    
    return jsonify({"status": "success"})

@app.route('/api/save-chat', methods=['POST'])
def save_chat():
    data = request.json
    chat_history = data.get('history', [])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    with open(f"chats/{filename}", 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)
    
    return jsonify({"status": "success", "filename": filename})


if __name__ == '__main__':
    # Ensure the `static` folder exists
    # Ensure the `chats` folder exists
    os.makedirs('chats', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
