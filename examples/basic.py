from ecliptor.client import Ecliptor
from openai import OpenAI

openai_client = OpenAI()

ecliptor_client = Ecliptor(
    api_key="5a2f36c0487885cc61afae69c4c4768087b86a0ab74652678be8e9415a9221157681adc4d52e5348cf2f1b43910235da",
    base_url="http://localhost:8000/" # todo update url - local testing only
)

def example_usage():
    # Example embedding from openai
    response = openai_client.embeddings.create(
        input="hello world",
        model="text-embedding-3-large"
    )

    query = response.data[0].embedding
    adapter_name = "residual_1"

    try:
        result = ecliptor_client.adapt(query, adapter_name)
        print("Adapted Embedding successfully calculated, length:", len(result))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    example_usage()
