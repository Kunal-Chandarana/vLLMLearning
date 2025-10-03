#!/usr/bin/env python3
"""
vLLM API Server Example

This example demonstrates how to create an OpenAI-compatible API server using vLLM.
Features:
- OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions)
- Streaming and non-streaming responses
- Multiple model support
- Custom middleware
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class VLLMServer:
    """vLLM API Server with OpenAI-compatible endpoints."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.engine = None
        self.app = FastAPI(title="vLLM API Server", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize the vLLM engine on startup."""
            await self.initialize_engine()
        
        @self.app.get("/")
        async def root():
            return {"message": "vLLM API Server", "model": self.model_name}
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "vllm",
                    }
                ]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint."""
            
            # Convert chat messages to a single prompt
            prompt = self.messages_to_prompt(request.messages)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )
            
            if request.stream:
                return StreamingResponse(
                    self.stream_chat_completion(prompt, sampling_params, request.model),
                    media_type="text/plain"
                )
            else:
                return await self.create_chat_completion(prompt, sampling_params, request.model)
        
        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """OpenAI-compatible completions endpoint."""
            
            # Handle both single prompt and list of prompts
            prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )
            
            if request.stream:
                if len(prompts) > 1:
                    raise HTTPException(status_code=400, detail="Streaming not supported for multiple prompts")
                return StreamingResponse(
                    self.stream_completion(prompts[0], sampling_params, request.model),
                    media_type="text/plain"
                )
            else:
                return await self.create_completion(prompts, sampling_params, request.model)
    
    async def initialize_engine(self):
        """Initialize the vLLM async engine."""
        print(f"🚀 Initializing vLLM engine with model: {self.model_name}")
        
        try:
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=1,
                max_model_len=512,
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            print("✅ vLLM engine initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize vLLM engine: {e}")
            raise
    
    def messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def create_chat_completion(self, prompt: str, sampling_params: SamplingParams, model: str):
        """Create a non-streaming chat completion response."""
        request_id = random_uuid()
        
        # Generate response
        results = []
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            results.append(output)
        
        # Get the final output
        final_output = results[-1]
        generated_text = final_output.outputs[0].text
        
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
            },
        }
    
    async def stream_chat_completion(self, prompt: str, sampling_params: SamplingParams, model: str):
        """Create a streaming chat completion response."""
        request_id = random_uuid()
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            for output_item in output.outputs:
                delta_text = output_item.text
                
                chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }
                    ],
                }
                
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def create_completion(self, prompts: List[str], sampling_params: SamplingParams, model: str):
        """Create a non-streaming completion response."""
        request_id = random_uuid()
        
        choices = []
        for i, prompt in enumerate(prompts):
            results = []
            async for output in self.engine.generate(prompt, sampling_params, f"{request_id}-{i}"):
                results.append(output)
            
            final_output = results[-1]
            generated_text = final_output.outputs[0].text
            
            choices.append({
                "index": i,
                "text": generated_text,
                "finish_reason": "stop",
            })
        
        return {
            "id": f"cmpl-{request_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
        }
    
    async def stream_completion(self, prompt: str, sampling_params: SamplingParams, model: str):
        """Create a streaming completion response."""
        request_id = random_uuid()
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            for output_item in output.outputs:
                chunk = {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": output_item.text,
                            "finish_reason": None,
                        }
                    ],
                }
                
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        yield "data: [DONE]\n\n"


def main():
    """Main function to run the server."""
    print("🌟 vLLM API Server Example")
    print("=" * 50)
    
    # You can change the model here
    model_name = "microsoft/DialoGPT-medium"
    
    # Create server instance
    server = VLLMServer(model_name=model_name)
    
    print(f"🚀 Starting server with model: {model_name}")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("\n💡 Example usage:")
    print("curl -X POST http://localhost:8000/v1/chat/completions \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"model": "microsoft/DialoGPT-medium", "messages": [{"role": "user", "content": "Hello!"}]}\'')
    
    # Run the server
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

