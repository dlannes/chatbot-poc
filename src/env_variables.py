from langchain_core.pydantic_v1 import SecretStr
import yaml
import os


def __load_prompt_templates() -> dict:
    with open("prompt_templates.yaml", "r") as file:
        return yaml.safe_load(file)


def __load_env_var(var_name: str) -> SecretStr:
    value = os.getenv(var_name)
    if not value:
        raise Exception(f"Environment variable '{var_name}' is not set.")
    return SecretStr(value)


prompt_templates: dict[str, str] = __load_prompt_templates()
OPENAI_API_KEY = __load_env_var("OPENAI_API_KEY")
VECTOR_DB_CONN = __load_env_var("VECTOR_DB_CONN_STR")
DATABASE_CONN = __load_env_var("DATABASE_CONN_STR")
