import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src import config

if not config.BEDROCK_MODEL_ID:
    raise SystemExit("BEDROCK_MODEL_ID is empty. Set it in your .env file.")

try:
    # boto3 finds your login in the standard AWS credential chain.
    # No keys appear anywhere in this code.
    session = boto3.Session(
        profile_name=config.AWS_PROFILE,
        region_name=config.AWS_REGION,
    )
    client = session.client("bedrock-runtime")

    response = client.converse(
        modelId=config.BEDROCK_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [{"text": "In one short sentence, what is Amazon Bedrock?"}],
        }],
        inferenceConfig={"maxTokens": 100, "temperature": 0},
    )

    print("Reply from Bedrock:")
    print(response["output"]["message"]["content"][0]["text"])

except NoCredentialsError:
    print("No AWS credentials found. Check: aws sts get-caller-identity")
except ClientError as error:
    code = error.response["Error"]["Code"]
    print("AWS error:", code)
    print(error.response["Error"]["Message"])