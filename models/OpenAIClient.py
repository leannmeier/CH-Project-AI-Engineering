from collections.abc import AsyncGenerator
from openai import AsyncOpenAI, APIError

from BaseLLMClient import BaseLLMClient
from Config import Config, ClientType
from exceptions import LLMClientError
from schemas import ChatMessage, FinishReason, ModelResponse, Usage


class OpenAIClient(BaseLLMClient):

    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=self.config.api_key.get_secret_value()
        )

    async def generate_response(self, messages: list[ChatMessage]) -> ModelResponse:
        openai_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except APIError as e:
            raise LLMClientError(
                message=e.message,
                provider=self.config.provider,
                status_code=getattr(e, "status_code", None),
                raw_error=e,
            ) from e

        choice = response.choices[0]

        finish_reason_map = {
            "stop": FinishReason.OK,
            "length": FinishReason.TOKEN_LIMIT,
            "tool_calls": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.CONTENT_FILTER,
        }

        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return ModelResponse(
            content=choice.message.content or "",
            provider=self.config.provider,
            model=response.model,
            usage=usage,
            finish_reason=finish_reason_map.get(
                choice.finish_reason, FinishReason.STOP
            ),
        )

    async def generate_response_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        openai_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIError as e:
            raise LLMClientError(
                message=e.message,
                provider=self.config.provider,
                status_code=getattr(e, "status_code", None),
                raw_error=e,
            ) from e
                
