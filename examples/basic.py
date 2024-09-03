from ..src.ecliptor import Ecliptor
from openai import OpenAI

openai_client = OpenAI()

ecliptor_client = Ecliptor(
    api_key="da9f4b90dafa2dba4ac25b70446db8135fd593f68613a8cb44c44e219ec7716f9e239b1d47ddf092d871ee194773ca38",
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
        result = ecliptor_client.adapt(query, adapter_name, 3072)
        print("Adapted Embedding successfully calculated, length:", len(result))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    example_usage()
