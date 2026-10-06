import uuid

import boto3

from src import config


def get_bedrock_llm():
    """
    Create a Bedrock Agent Runtime client.

    The Bedrock Agent is responsible for calling the
    configured foundation model, such as Amazon Nova.
    """

    if not config.BEDROCK_AGENT_ID:
        raise RuntimeError(
            "BEDROCK_AGENT_ID is empty. "
            "Set it in your .env file."
        )

    if not config.BEDROCK_AGENT_ALIAS_ID:
        raise RuntimeError(
            "BEDROCK_AGENT_ALIAS_ID is empty. "
            "Set it in your .env file."
        )

    if not config.AWS_REGION:
        raise RuntimeError(
            "AWS_REGION is empty. "
            "Set it in your .env file."
        )

    session = boto3.Session(
        profile_name=config.AWS_PROFILE
        if config.AWS_PROFILE
        else None
    )

    client = session.client(
        "bedrock-agent-runtime",
        region_name=config.AWS_REGION,
    )

    return {
        "client": client,
        "agent_id": config.BEDROCK_AGENT_ID,
        "agent_alias_id": config.BEDROCK_AGENT_ALIAS_ID,
    }


def invoke_bedrock_agent(agent, prompt):
    """
    Send the RAG prompt to the Amazon Bedrock Agent.

    The Agent then uses its configured foundation model,
    such as Amazon Nova, to generate the final response.
    """

    session_id = str(uuid.uuid4())

    response = agent["client"].invoke_agent(
        agentId=agent["agent_id"],
        agentAliasId=agent["agent_alias_id"],
        sessionId=session_id,
        inputText=prompt,
    )

    answer = ""

    for event in response.get("completion", []):
        chunk = event.get("chunk")

        if chunk and "bytes" in chunk:
            answer += chunk["bytes"].decode("utf-8")

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            "Amazon Bedrock Agent returned an empty response."
        )

    return answer


def check_bedrock_llm_ready(agent):
    """
    Verify that the Bedrock Agent can be reached.

    This performs a small test request.
    """

    try:
        invoke_bedrock_agent(
            agent,
            "Reply with the word READY.",
        )

    except Exception as error:

        message = str(error)

        if (
            "credentials" in message.lower()
            or "NoCredentials" in message
        ):
            raise RuntimeError(
                "AWS credentials not found. "
                "Run: aws sts get-caller-identity"
            )

        if "AccessDenied" in message:
            raise RuntimeError(
                "AWS denied access to the Bedrock Agent. "
                "Check your IAM permissions and Agent configuration."
            )

        if (
            "ResourceNotFound" in message
            or "ValidationException" in message
        ):
            raise RuntimeError(
                "Bedrock Agent could not be found. "
                f"Check BEDROCK_AGENT_ID='{agent['agent_id']}', "
                f"BEDROCK_AGENT_ALIAS_ID='{agent['agent_alias_id']}', "
                f"and AWS_REGION."
            )

        raise RuntimeError(
            f"Could not reach Amazon Bedrock Agent: {message}"
        )