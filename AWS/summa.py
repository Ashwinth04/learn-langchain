import boto3
from langchain_community.embeddings import BedrockEmbeddings

# Initialize the Bedrock client

bedrock = boto3.client(service_name = "bedrock-runtime",region_name='us-east-2')
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1",client = bedrock)
# Your text data
text_data = ["Sample text 1", "Sample text 2"]

# Create embeddings
response = bedrock_embeddings(
    Input={
        'TextList': text_data
    }
)

embeddings = response['Embeddings']
print(embeddings)
