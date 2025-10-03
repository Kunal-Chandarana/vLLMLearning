#!/usr/bin/env python3
"""
Streaming Response Example

This example demonstrates how to use vLLM for streaming text generation.
Features:
- Real-time token streaming
- Progress indicators
- Async streaming
- Multiple concurrent streams
"""

import asyncio
import time
from typing import AsyncGenerator, List
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


class StreamingExample:
    """Demonstrates streaming capabilities of vLLM."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.engine = None
    
    async def initialize_engine(self):
        """Initialize the async vLLM engine."""
        print(f"🚀 Initializing async engine with {self.model_name}...")
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=1,
            max_model_len=512,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✅ Engine initialized!")
    
    async def stream_single_response(self, prompt: str, sampling_params: SamplingParams) -> AsyncGenerator[str, None]:
        """Stream a single response token by token."""
        request_id = random_uuid()
        
        print(f"🎯 Streaming response for: '{prompt[:50]}...'")
        print("📝 Generated text: ", end="", flush=True)
        
        full_text = ""
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                new_text = output.outputs[0].text
                # Only yield the new part (delta)
                delta = new_text[len(full_text):]
                if delta:
                    print(delta, end="", flush=True)
                    full_text = new_text
                    yield delta
        
        print("\n✅ Streaming completed!")
    
    async def demonstrate_streaming(self):
        """Demonstrate various streaming scenarios."""
        
        prompts = [
            "The future of artificial intelligence will",
            "In the year 2030, technology will have",
            "The most important lesson I learned is",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=80,
            repetition_penalty=1.1,
        )
        
        print("\n🌊 Single Stream Demonstration")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts):
            print(f"\n📋 Example {i+1}:")
            
            # Collect all streamed tokens
            streamed_tokens = []
            async for token in self.stream_single_response(prompt, sampling_params):
                streamed_tokens.append(token)
            
            print(f"📊 Total tokens streamed: {len(streamed_tokens)}")
            await asyncio.sleep(1)  # Brief pause between examples
    
    async def demonstrate_concurrent_streaming(self):
        """Demonstrate multiple concurrent streams."""
        print("\n🚀 Concurrent Streaming Demonstration")
        print("=" * 60)
        
        prompts = [
            "The benefits of renewable energy include",
            "Artificial intelligence can help solve",
            "The key to effective communication is",
            "In a sustainable future, we will",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=60,
        )
        
        # Start all streams concurrently
        tasks = []
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(
                self.stream_with_id(prompt, sampling_params, f"Stream-{i+1}")
            )
            tasks.append(task)
        
        # Wait for all streams to complete
        results = await asyncio.gather(*tasks)
        
        print(f"\n✅ All {len(results)} concurrent streams completed!")
    
    async def stream_with_id(self, prompt: str, sampling_params: SamplingParams, stream_id: str):
        """Stream with identification for concurrent demonstration."""
        request_id = random_uuid()
        
        print(f"\n🎯 {stream_id}: Starting stream for '{prompt[:30]}...'")
        
        start_time = time.time()
        token_count = 0
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                token_count += 1
        
        duration = time.time() - start_time
        print(f"✅ {stream_id}: Completed in {duration:.2f}s ({token_count} tokens)")
        
        return {
            "stream_id": stream_id,
            "duration": duration,
            "tokens": token_count,
            "tokens_per_second": token_count / duration if duration > 0 else 0
        }
    
    async def demonstrate_streaming_with_callbacks(self):
        """Demonstrate streaming with custom callbacks."""
        print("\n🔄 Streaming with Callbacks")
        print("=" * 60)
        
        prompt = "The evolution of machine learning has led to"
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=100,
        )
        
        # Callback functions
        def on_token(token: str, position: int):
            if position % 10 == 0:  # Every 10th token
                print(f"\n[Token {position}] ", end="")
            print(token, end="", flush=True)
        
        def on_complete(total_tokens: int, duration: float):
            print(f"\n\n📊 Generation complete!")
            print(f"   Total tokens: {total_tokens}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Speed: {total_tokens/duration:.1f} tokens/sec")
        
        # Stream with callbacks
        await self.stream_with_callbacks(prompt, sampling_params, on_token, on_complete)
    
    async def stream_with_callbacks(self, prompt: str, sampling_params: SamplingParams, 
                                  on_token_callback, on_complete_callback):
        """Stream with custom callback functions."""
        request_id = random_uuid()
        
        start_time = time.time()
        token_count = 0
        full_text = ""
        
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                new_text = output.outputs[0].text
                delta = new_text[len(full_text):]
                
                if delta:
                    on_token_callback(delta, token_count)
                    token_count += 1
                    full_text = new_text
        
        duration = time.time() - start_time
        on_complete_callback(token_count, duration)


async def main():
    """Main function to run streaming examples."""
    print("🌊 vLLM Streaming Response Example")
    print("=" * 50)
    
    # Initialize streaming example
    example = StreamingExample()
    await example.initialize_engine()
    
    try:
        # Run different streaming demonstrations
        await example.demonstrate_streaming()
        await example.demonstrate_concurrent_streaming()
        await example.demonstrate_streaming_with_callbacks()
        
        print("\n🎉 All streaming examples completed!")
        print("\n💡 Key takeaways:")
        print("   - vLLM supports real-time token streaming")
        print("   - Multiple concurrent streams are possible")
        print("   - Callbacks enable custom processing")
        print("   - Async/await pattern provides efficient handling")
        
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
    
    finally:
        # Cleanup would go here if needed
        pass


if __name__ == "__main__":
    asyncio.run(main())

