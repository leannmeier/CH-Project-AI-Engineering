from collections.abc import AsyncGenerator
from anthropic import AsyncAnthropic, APIError

from BaseLLMClient import BaseLLMClient
from Config import Config, ClientType
from exceptions import LLMClientError
from schemas import ChatMessage, FinishReason, ModelResponse, Usage


class AnthropicClient(BaseLLMClient):

    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncAnthropic(
            api_key=self.config.api_key.get_secret_value()
        )

    async def generate_response(self, messages: list[ChatMessage]) -> ModelResponse:
        anthropic_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        try:
            response = await self.client.messages.create(
                max_tokens=self.config.max_tokens,
                messages=anthropic_messages,
                model=self.config.model,
                )
        except APIError as e:
            raise LLMClientError(
                message=e.message,
                provider=self.config.provider,
                status_code=getattr(e, "status_code", None),
                raw_error=e,
            ) from e

        finish_reason_map = {
            "end_turn": FinishReason.OK,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.TOKEN_LIMIT,
            "tool_use": FinishReason.TOOL_CALL,
        }

        return ModelResponse(
            content=response.content[0].text,
            provider=self.config.provider,
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=finish_reason_map.get(
                response.stop_reason, FinishReason.STOP
            ),
        )

    async def generate_response_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        anthropic_messages = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]

        try:
            async with self.client.messages.stream(
                max_tokens=self.config.max_tokens,
                messages=anthropic_messages,
                model=self.config.model,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except APIError as e:
            raise LLMClientError(
                message=e.message,
                provider=self.config.provider,
                status_code=getattr(e, "status_code", None),
                raw_error=e,
            ) from e