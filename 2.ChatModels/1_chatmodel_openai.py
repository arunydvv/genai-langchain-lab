from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Temperature = 0.0
llm_low = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.0,
    max_completion_tokens=10
)

result_low = llm_low.invoke(
    "Write a creative one-line description of a sunset."
)

print("Temperature 0.0:")
print(result_low)


# Temperature = 1.8
llm_high = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=1.8
)

result_high = llm_high.invoke(
    "Write a creative one-line description of a sunset."
)

print("\nTemperature 1.8:")
print(result_high)