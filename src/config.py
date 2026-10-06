import os

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1",
)

AWS_PROFILE = os.getenv(
    "AWS_PROFILE",
    "",
)

BEDROCK_AGENT_ID = os.getenv(
    "BEDROCK_AGENT_ID",
    "",
)

BEDROCK_AGENT_ALIAS_ID = os.getenv(
    "BEDROCK_AGENT_ALIAS_ID",
    "",
)