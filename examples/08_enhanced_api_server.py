#!/usr/bin/env python3
"""
Enhanced vLLM API Server with Token Decoding

This example demonstrates an enhanced OpenAI-compatible API server using vLLM with token decoding features.
Features:
- OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions)
- Token decoding information in responses
- Streaming and non-streaming responses with token details
- Debug endpoints for token analysis
- Enhanced error handling
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

# Transformers for tokenizer
from transformers import AutoTokenizer


# Enhanced Pydantic models for API requests/responses
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
    # Enhanced: Include token information in response
    include_tokens: Optional[bool] = False
    # Enhanced: Include individual token breakdown
    include_token_breakdown: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 100
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    # Enhanced: Include token information in response
    include_tokens: Optional[bool] = False
    # Enhanced: Include individual token breakdown
    include_token_breakdown: Optional[bool] = False


class TokenDecodeRequest(BaseModel):
    """Request model for token decoding endpoint."""
    token_ids: List[int]
    skip_special_tokens: Optional[bool] = True
    clean_up_tokenization_spaces: Optional[bool] = True


class EnhancedVLLMServer:
    """Enhanced vLLM API Server with token decoding capabilities."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.engine = None
        self.tokenizer = None
        self.app = FastAPI(
            title="Enhanced vLLM API Server with Token Decoding", 
            version="2.0.0",
            description="OpenAI-compatible API server with enhanced token decoding features"
        )
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
            """Initialize the vLLM engine and tokenizer on startup."""
            await self.initialize_engine()
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Enhanced vLLM API Server with Token Decoding",
                "model": self.model_name,
                "features": [
                    "OpenAI-compatible endpoints",
                    "Token decoding information",
                    "Streaming responses",
                    "Debug endpoints"
                ]
            }
        
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
                        "capabilities": [
                            "text_completion",
                            "chat_completion",
                            "token_decoding"
                        ]
                    }
                ]
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Enhanced OpenAI-compatible chat completions endpoint with token decoding."""
            
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
                    self.stream_chat_completion(
                        prompt, sampling_params, request.model, 
                        request.include_tokens, request.include_token_breakdown
                    ),
                    media_type="text/plain"
                )
            else:
                return await self.create_chat_completion(
                    prompt, sampling_params, request.model,
                    request.include_tokens, request.include_token_breakdown
                )
        
        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            """Enhanced OpenAI-compatible completions endpoint with token decoding."""
            
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
                    self.stream_completion(
                        prompts[0], sampling_params, request.model,
                        request.include_tokens, request.include_token_breakdown
                    ),
                    media_type="text/plain"
                )
            else:
                return await self.create_completion(
                    prompts, sampling_params, request.model,
                    request.include_tokens, request.include_token_breakdown
                )
        
        # Enhanced: Token decoding endpoints
        @self.app.post("/v1/tokens/decode")
        async def decode_tokens(request: TokenDecodeRequest):
            """Decode token IDs back to text."""
            try:
                decoded_text = self.tokenizer.decode(
                    request.token_ids,
                    skip_special_tokens=request.skip_special_tokens,
                    clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                )
                
                # Individual token breakdown
                individual_tokens = []
                for i, token_id in enumerate(request.token_ids):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=request.skip_special_tokens)
                    individual_tokens.append({
                        "index": i,
                        "token_id": token_id,
                        "text": token_text
                    })
                
                return {
                    "decoded_text": decoded_text,
                    "token_count": len(request.token_ids),
                    "individual_tokens": individual_tokens,
                    "parameters": {
                        "skip_special_tokens": request.skip_special_tokens,
                        "clean_up_tokenization_spaces": request.clean_up_tokenization_spaces
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Token decoding failed: {str(e)}")
        
        @self.app.post("/v1/tokens/encode")
        async def encode_text(text: str = Query(..., description="Text to encode into tokens")):
            """Encode text into token IDs."""
            try:
                token_ids = self.tokenizer.encode(text)
                
                # Individual token breakdown
                individual_tokens = []
                for i, token_id in enumerate(token_ids):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    individual_tokens.append({
                        "index": i,
                        "token_id": token_id,
                        "text": token_text
                    })
                
                return {
                    "text": text,
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                    "individual_tokens": individual_tokens
                }
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Text encoding failed: {str(e)}")
        
        @self.app.get("/v1/tokens/special")
        async def get_special_tokens():
            """Get information about special tokens."""
            special_tokens = {}
            
            # Common special tokens
            token_attrs = ['eos_token_id', 'pad_token_id', 'unk_token_id', 'bos_token_id']
            
            for attr in token_attrs:
                token_id = getattr(self.tokenizer, attr, None)
                if token_id is not None:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    special_tokens[attr] = {
                        "id": token_id,
                        "text": token_text,
                        "description": attr.replace('_', ' ').title()
                    }
            
            return {
                "model": self.model_name,
                "special_tokens": special_tokens,
                "vocab_size": self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else None
            }
    
    async def initialize_engine(self):
        """Initialize the vLLM async engine and tokenizer."""
        print(f"🚀 Initializing enhanced vLLM engine with model: {self.model_name}")
        
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
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Enhanced vLLM engine and tokenizer initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize enhanced vLLM engine: {e}")
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
    
    def add_token_info(self, response: dict, final_output, include_tokens: bool, include_token_breakdown: bool):
        """Add token information to response."""
        if not include_tokens and not include_token_breakdown:
            return response
        
        # Get token information
        generated_token_ids = final_output.outputs[0].token_ids
        prompt_token_ids = final_output.prompt_token_ids
        
        if include_tokens:
            # Add basic token information
            response["token_info"] = {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(generated_token_ids),
                "total_tokens": len(prompt_token_ids) + len(generated_token_ids),
                "completion_token_ids": generated_token_ids
            }
            
            # Add decoded verification
            if self.tokenizer:
                decoded_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                response["token_info"]["decoded_verification"] = decoded_text
        
        if include_token_breakdown:
            # Add detailed token breakdown
            individual_tokens = []
            for i, token_id in enumerate(generated_token_ids):
                if self.tokenizer:
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    token_raw = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    individual_tokens.append({
                        "index": i,
                        "token_id": token_id,
                        "text": token_text,
                        "raw": token_raw
                    })
            
            response["token_breakdown"] = individual_tokens
        
        return response
    
    async def create_chat_completion(self, prompt: str, sampling_params: SamplingParams, model: str, 
                                   include_tokens: bool = False, include_token_breakdown: bool = False):
        """Create an enhanced non-streaming chat completion response."""
        request_id = random_uuid()
        
        # Generate response
        results = []
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            results.append(output)
        
        # Get the final output
        final_output = results[-1]
        generated_text = final_output.outputs[0].text
        
        response = {
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
        
        # Add enhanced token information
        return self.add_token_info(response, final_output, include_tokens, include_token_breakdown)
    
    async def stream_chat_completion(self, prompt: str, sampling_params: SamplingParams, model: str,
                                   include_tokens: bool = False, include_token_breakdown: bool = False):
        """Create an enhanced streaming chat completion response."""
        request_id = random_uuid()
        accumulated_tokens = []
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            for output_item in output.outputs:
                # Get new tokens (delta)
                current_tokens = output_item.token_ids
                new_tokens = current_tokens[len(accumulated_tokens):]
                
                # Calculate delta text
                if self.tokenizer and new_tokens:
                    delta_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                else:
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
                
                # Add token information to streaming chunks
                if include_tokens and new_tokens:
                    chunk["token_info"] = {
                        "new_tokens": new_tokens,
                        "accumulated_tokens": len(current_tokens),
                        "delta_text": delta_text
                    }
                
                if include_token_breakdown and new_tokens:
                    chunk["token_breakdown"] = [
                        {
                            "token_id": token_id,
                            "text": self.tokenizer.decode([token_id], skip_special_tokens=True) if self.tokenizer else ""
                        }
                        for token_id in new_tokens
                    ]
                
                accumulated_tokens = current_tokens
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
        
        if include_tokens:
            final_chunk["token_info"] = {
                "total_completion_tokens": len(accumulated_tokens),
                "final_tokens": accumulated_tokens
            }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    async def create_completion(self, prompts: List[str], sampling_params: SamplingParams, model: str,
                              include_tokens: bool = False, include_token_breakdown: bool = False):
        """Create an enhanced non-streaming completion response."""
        request_id = random_uuid()
        
        choices = []
        for i, prompt in enumerate(prompts):
            results = []
            async for output in self.engine.generate(prompt, sampling_params, f"{request_id}-{i}"):
                results.append(output)
            
            final_output = results[-1]
            generated_text = final_output.outputs[0].text
            
            choice = {
                "index": i,
                "text": generated_text,
                "finish_reason": "stop",
            }
            
            # Add token information to individual choices
            if include_tokens:
                token_ids = final_output.outputs[0].token_ids
                choice["token_info"] = {
                    "completion_tokens": len(token_ids),
                    "token_ids": token_ids
                }
                
                if self.tokenizer:
                    choice["token_info"]["decoded_verification"] = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            
            if include_token_breakdown and self.tokenizer:
                token_ids = final_output.outputs[0].token_ids
                choice["token_breakdown"] = [
                    {
                        "index": j,
                        "token_id": token_id,
                        "text": self.tokenizer.decode([token_id], skip_special_tokens=True)
                    }
                    for j, token_id in enumerate(token_ids)
                ]
            
            choices.append(choice)
        
        response = {
            "id": f"cmpl-{request_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
        }
        
        # Add overall usage information
        if choices:
            total_completion_tokens = sum(len(final_output.outputs[0].token_ids) for final_output in [results[-1]])
            response["usage"] = {
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_completion_tokens  # Prompt tokens would need to be calculated per prompt
            }
        
        return response
    
    async def stream_completion(self, prompt: str, sampling_params: SamplingParams, model: str,
                              include_tokens: bool = False, include_token_breakdown: bool = False):
        """Create an enhanced streaming completion response."""
        request_id = random_uuid()
        accumulated_tokens = []
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            for output_item in output.outputs:
                # Get new tokens (delta)
                current_tokens = output_item.token_ids
                new_tokens = current_tokens[len(accumulated_tokens):]
                
                # Calculate delta text
                if self.tokenizer and new_tokens:
                    delta_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                else:
                    delta_text = output_item.text
                
                chunk = {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": delta_text,
                            "finish_reason": None,
                        }
                    ],
                }
                
                # Add token information to streaming chunks
                if include_tokens and new_tokens:
                    chunk["token_info"] = {
                        "new_tokens": new_tokens,
                        "accumulated_tokens": len(current_tokens),
                        "delta_text": delta_text
                    }
                
                if include_token_breakdown and new_tokens:
                    chunk["token_breakdown"] = [
                        {
                            "token_id": token_id,
                            "text": self.tokenizer.decode([token_id], skip_special_tokens=True) if self.tokenizer else ""
                        }
                        for token_id in new_tokens
                    ]
                
                accumulated_tokens = current_tokens
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        yield "data: [DONE]\n\n"


def main():
    """Main function to run the enhanced server."""
    print("🌟 Enhanced vLLM API Server with Token Decoding")
    print("=" * 60)
    
    # Using GPT-2 for better reliability and token decoding demonstration
    model_name = "gpt2"
    
    # Create server instance
    server = EnhancedVLLMServer(model_name=model_name)
    
    print(f"🚀 Starting enhanced server with model: {model_name}")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("\n💡 Enhanced features:")
    print("   - Token decoding in completion responses")
    print("   - Debug endpoints for token analysis")
    print("   - Streaming with token information")
    print("\n🔤 Token endpoints:")
    print("   POST /v1/tokens/decode - Decode token IDs to text")
    print("   POST /v1/tokens/encode - Encode text to token IDs")
    print("   GET  /v1/tokens/special - Get special token information")
    print("\n💡 Example usage with token decoding:")
    print("curl -X POST http://localhost:8000/v1/completions \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"model": "gpt2", "prompt": "Hello world", "include_tokens": true, "include_token_breakdown": true}\'')
    
    # Run the server
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
