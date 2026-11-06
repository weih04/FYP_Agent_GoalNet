"""Model calls. Any OpenAI-compatible endpoint that supports json_schema response formats
(OpenAI, vLLM, Ollama, a hosted Llama/Qwen server) runs through the same code, so model
families and sizes are compared under one code path."""

import os

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


def _load_key(env_var, secrets_path):
    if os.environ.get(env_var):
        return
    if secrets_path and os.path.exists(secrets_path):
        with open(secrets_path) as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == env_var and value:
                    os.environ[env_var] = value
                    return
    raise RuntimeError(f"{env_var} is neither set nor in {secrets_path}")


def call_model(prompt, model, temperature, response_format, base_url=None, api_key_env="OPENAI_API_KEY",
               secrets_path=None):
    """One structured-output call. Returns (raw text, input tokens, output tokens, served model)."""
    from openai import OpenAI

    _load_key(api_key_env, secrets_path)
    client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
    resp = client.chat.completions.parse(
        model=model,
        temperature=temperature,
        response_format=response_format,
        messages=[{"role": "user", "content": prompt}],
    )
    msg = resp.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"model refused: {msg.refusal}")
    return msg.content, resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.model
