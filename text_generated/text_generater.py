from openai import OpenAI

class DescriptionGenerator:
    def __init__(self, api_key, base_url):
        # 初始化客户端，指向硅基流动 API（与原来一致）
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    # 纯文本问答（已构建好的 prompt 直接传入）
    def get_text_answer(self, prompt, system=None, temperature=0.7, max_tokens=None):
        """
        使用 deepseek v3 进行纯文本对话，返回最终答案字符串。
        - prompt: 你已经构建好的纯文本提示词
        - system: 可选的系统提示（如需约束风格/角色）
        - temperature: 采样温度
        - max_tokens: 生成上限（不传则由服务端默认）
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model="deepseek-ai/deepseek-v3",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content

   