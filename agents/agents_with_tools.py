from dotenv import load_dotenv
import os
from dataclasses import dataclass

from pydantic import BaseModel
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from tools.system_metrics import gpu_util_metrics2df

load_dotenv()

# ==============================================================================
# CONFIG
# ==============================================================================
DEV = True

CITY = "Johannesburg"
GPU_CMD = (
    "nvidia-smi --format=csv "
    "--query-gpu=power.draw,fan.speed,utilization.gpu,"
    "temperature.gpu,memory.used,memory.total"
)

# ==============================================================================
# SCHEMAS
# ==============================================================================
class Weather(BaseModel):
    city: str


class Metrics(BaseModel):
    cmd: str


@dataclass
class WeatherContext:
    city: str


@dataclass
class MetricsContext:
    cmd: str


# ==============================================================================
# TOOLS
# ==============================================================================
@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city using Google scraping.
    """
    import json
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    city_query = city.replace(" ", "+")
    url = f"https://www.google.com/search?q={city_query}+weather"

    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    location = soup.select_one("#wob_loc").get_text(strip=True)
    time = soup.select_one("#wob_dts").get_text(strip=True)
    info = soup.select_one("#wob_dc").get_text(strip=True)
    temp = soup.select_one("#wob_tm").get_text(strip=True)

    return json.dumps(
        {
            "location": location,
            "time": time,
            "info": info,
            "weather": f"{temp}°C",
        }
    )


@tool
def get_gpu_metrics(cmd: str):
    """
    Run nvidia-smi and return GPU utilization metrics.
    """
    return gpu_util_metrics2df(cmd)


# ==============================================================================
# SYSTEM PROMPTS
# ==============================================================================
SYSTEM_PROMPT_WEATHER = """You are a concise agent with tool access.
Use `get_weather` to answer weather questions.
"""

SYSTEM_PROMPT_GPU = """You are a concise agent with tool access.
Use `get_gpu_metrics` to answer GPU metric questions.
"""


# ==============================================================================
# RUNTIME
# ==============================================================================
if DEV:
    # --------------------------------------------------------------------------
    # DEV MODE: Ollama / OpenAI-compatible client (NO ToolStrategy)
    # --------------------------------------------------------------------------
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",  # required but ignored by Ollama
    )

    response = client.chat.completions.create(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": f"What's the weather right now in {CITY}?",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": Weather.model_json_schema(),
        },
    )

    print("\n=== RAW RESPONSE ===")
    print(response.choices[0].message)

else:
    # --------------------------------------------------------------------------
    # PROD MODE: LangChain Agent (ToolStrategy OK)
    # --------------------------------------------------------------------------
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=800,
        timeout=30,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # --------------------
    # Weather Agent
    # --------------------
    weather_agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT_WEATHER,
        response_format=ToolStrategy(Weather),
        context_schema=WeatherContext,
    )

    weather_result = weather_agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
        context=WeatherContext(city=CITY),
    )

    print("\n=== WEATHER RESULT ===")
    print(weather_result)

    # --------------------
    # GPU Agent (optional)
    # --------------------
    """
    gpu_agent = create_agent(
        model=model,
        tools=[get_gpu_metrics],
        system_prompt=SYSTEM_PROMPT_GPU,
        response_format=ToolStrategy(Metrics),
        context_schema=MetricsContext,
    )

    gpu_result = gpu_agent.invoke(
        {"messages": [{"role": "user", "content": "What are the GPU metrics?"}]},
        context=MetricsContext(cmd=GPU_CMD),
    )

    print("\n=== GPU RESULT ===")
    print(gpu_result)
    """